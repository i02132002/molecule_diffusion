import pims
import pySPM as spm
import numpy as np
from skimage.transform import resize
from skimage import exposure

#@pims.pipeline
#def gray(image):
#    return image[:, :, 1]  # Take just the green channel
#frames = gray(pims.open('../track_molecules/*.bmp'))

class SXMReader(pims.FramesSequence):
    def __init__(self, filename_pattern, channel = "Z", correct = None):
        self.filenames = filename_pattern
        self.scans = [spm.SXM(filename) for filename in self.filenames]
        self.smallest = 9999999
        self.z_data = []
        self.channel = channel
        for i, s in enumerate(self.scans):
            if channel == "Bias":
                pxs = s.get_channel("Bias").pixels
            else:
                if correct == "lines":
                    pxs = s.get_channel("Z").correct_lines().pixels
                elif correct == "plane":
                    pxs = s.get_channel("Z").correct_plane().pixels
                elif correct == "slope":
                    pxs = s.get_channel("Z").correct_slope().pixels
                else:
                    pxs = s.get_channel("Z").pixels
            
            f = open(self.filenames[i], "r", encoding='latin-1')
            lines = f.readlines()
            scan_up = [lines[i+1 % len(lines)] for i, x in enumerate(lines) if x == ':SCAN_DIR:\n'][0].strip() == 'up'
            if not scan_up:
                pxs = pxs[::-1]
            
            self.smallest = min(self.smallest, pxs.shape[0])
            self.z_data.append(pxs)
        self._len = len(self.z_data)
        self._dtype = np.float32
        self._frame_shape = (self.smallest, self.smallest)
        self.scan_size = self.scans[0].size
        self.meters_per_pixel = self.scan_size['real']['x']/ self.smallest
       
    def get_frame(self, i):
        # Access the data you need and get it into a numpy array.
        # Then return a Frame like so:
        image = self.z_data[i]
        if self.channel == "Bias":
            return pims.Frame(image, frame_no=i)
        else:
            v_min, v_max = np.percentile(image, (0.2, 99.8))
            image = exposure.rescale_intensity(
                image, in_range=(v_min, v_max)
                )        
#             image = resize(image, self.frame_shape)
            return pims.Frame(image, frame_no=i)
    
    def __len__(self):
        return self._len
    
    def __setitem__(self, key, value):
        self.z_data[key] = value

    @property
    def frame_shape(self):
        return self._frame_shape

    @property
    def pixel_type(self):
        return self._dtype