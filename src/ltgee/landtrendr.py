import ee
from datetime import datetime

class Sentinel2LandTrendr:
    def __init__(self, collection_name='COPERNICUS/S2_SR'):
        self.collection_name = collection_name
        self.collection = ee.ImageCollection(self.collection_name)\
                             .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))  # Filter cloud cover to 20% or less

    def run_landtrendr(self, params):
        """
        Runs LandTrendr segmentation on a Sentinel-2 time series.
        """
        lt = ee.Algorithms.TemporalSegmentation.LandTrendr(**params)
        return lt

    def get_spectral_index(self, index_name):
        """
        Retrieves the specified spectral index for Sentinel-2.
        """
        def add_index(img):
            # Add any custom spectral indices here if needed
            if index_name == 'NDVI':
                ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
                return img.addBands(ndvi)
            else:
                raise ValueError(f"Index '{index_name}' is not implemented for Sentinel-2.")
        
        return self.collection.map(add_index).select(index_name)

    def apply_mmu(self, image: ee.Image, mmu_value: int) -> ee.Image:
        """
        Applies a minimum mapping unit (MMU) filter to the image based on connected pixel count.
        """
        mmu_image = image.select([0])\
            .gte(ee.Number(1))\
            .selfMask()\
            .connectedPixelCount()
        min_area = mmu_image.gte(ee.Number(mmu_value)).selfMask()
        return min_area.reproject(image.projection().atScale(10)).unmask()

    def get_fitted_data(self, index: str, start_date: datetime, end_date: datetime) -> ee.Image:
        """
        Generates an annual band stack for a specified index with fitted-to-vertex values.
        """
        search = 'ftv_' + index.lower() + '_fit'
        return self.collection.select(search)\
                    .arrayFlatten([[str(y) for y in range(start_date.year, end_date.year + 1)]])

    def get_segment_data(self, delta: str, index_flip: bool, options=None) -> ee.Image:
        """
        Retrieves segment data with parameters for delta type ('all', 'gain', 'loss') and options for orientation.
        """
        lt_band = self.select('LandTrendr')
        rmse = self.select('rmse')
        vertex_mask = lt_band.arraySlice(0, 3, 4)
        vertices = lt_band.arrayMask(vertex_mask)

        left_list = vertices.arraySlice(1, 0, -1)
        right_list = vertices.arraySlice(1, 1, None)
        start_year = left_list.arraySlice(0, 0, 1)
        start_val = left_list.arraySlice(0, 2, 3)
        end_year = right_list.arraySlice(0, 0, 1)
        end_val = right_list.arraySlice(0, 2, 3)
        dur = end_year.subtract(start_year)
        mag = end_val.subtract(start_val)
        rate = mag.divide(dur)
        dsnr = mag.divide(rmse)

        if delta == 'all':
            if options and options.get('right'):
                if index_flip:
                    start_val = start_val.multiply(-1)
                    end_val = end_val.multiply(-1)
                    mag = mag.multiply(-1)
                    rate = rate.multiply(-1)
                    dsnr = dsnr.multiply(-1)

            return ee.Image.cat([start_year, end_year, start_val, end_val, mag, dur, rate, dsnr]) \
                .unmask(ee.Image(ee.Array([[-9999]]))) \
                .toArray(0)

        elif delta in ['gain', 'loss']:
            change_type_mask = mag.lt(0) if delta == 'gain' else mag.gt(0)
            flip = -1 if index_flip else 1
            return ee.Image.cat([
                start_year.arrayMask(change_type_mask),
                end_year.arrayMask(change_type_mask),
                start_val.arrayMask(change_type_mask).multiply(flip),
                end_val.arrayMask(change_type_mask).multiply(flip),
                mag.arrayMask(change_type_mask).abs(),
                dur.arrayMask(change_type_mask),
                rate.arrayMask(change_type_mask).abs(),
                dsnr.arrayMask(change_type_mask).abs(),
            ]).unmask(ee.Image(ee.Array([[-9999]]))).toArray(0)

    def collection_to_band_stack(self, collection: ee.ImageCollection, start_date: datetime, end_date: datetime, mask_fill: int = 0) -> ee.Image:
        """
        Transforms an image collection into a stacked band image.
        """
        unmasked_collection = collection.map(lambda image: image.unmask(mask_fill))
        collection_array = unmasked_collection.toArrayPerBand()
        bands = unmasked_collection.first().bandNames().getInfo()
        all_stack = ee.Image()

        for band in bands:
            band_ts = collection_array.select(band).arrayFlatten(
                [[str(y) for y in range(start_date.year, end_date.year + 1)]]
            )
            all_stack = ee.Image.cat([all_stack, band_ts])

        return all_stack.slice(1, None).toUint16()

    def get_fitted_rgb_col(self, bands: list[str], vis_params: dict, start_date: datetime, end_date: datetime) -> ee.ImageCollection:
        """
        Creates an RGB image collection from fitted data for visualization.
        """
        r = self.get_fitted_data(bands[0], start_date, end_date)
        g = self.get_fitted_data(bands[1], start_date, end_date)
        b = self.get_fitted_data(bands[2], start_date, end_date)
        rgb_list = []

        for year in range(start_date.year, end_date.year + 1):
            year_str = str(year)
            rgb_list.append(r.select(year_str)
                            .addBands(g.select(year_str))
                            .addBands(b.select(year_str))
                            .rename(['R', 'G', 'B']))

        rgb_col = ee.ImageCollection(rgb_list)\
            .map(lambda image: image.visualize(**vis_params))\
            .map(lambda image: image.set({
                'system:time_start': ee.Date.fromYMD(start_date.year, start_date.month, start_date.day).millis(),
                'composite_year': start_date.year
            }))

        return rgb_col

    def get_segment_count(self, segment_data: ee.Image) -> ee.Image:
        """
        Counts the number of segments in the LandTrendr output.
        """
        return segment_data.arrayLength(1).select([0], ['segCount']).toByte()

    def getLTvertStack(self, lt, runParams):
        """
        Extracts vertices from LandTrendr results and creates a band stack for the vertices.
        """
        lt = lt.select('LandTrendr')
        emptyArray = []  # Empty array to hold another array whose length will vary depending on maxSegments parameter
        vertLabels = []  # Empty array to hold band names whose length will vary depending on maxSegments parameter

        for i in range(1, runParams['maxSegments'] + 2):
            vertLabels.append("vert_" + str(i))
            emptyArray.append(0)

        zeros = ee.Image(ee.Array([emptyArray, emptyArray, emptyArray]))
        lbls = [['yrs_', 'src_', 'fit_'], vertLabels]
        vmask = lt.arraySlice(0, 3, 4)

        ltVertStack = (lt.arrayMask(vmask)
                         .arraySlice(0, 0, 3)
                         .addBands(zeros)
                         .toArray(1)
                         .arraySlice(1, 0, runParams['maxSegments'] + 1)
                         .arrayFlatten(lbls, ''))

        return ltVertStack
