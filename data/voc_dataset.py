import os
import xml.etree.ElementTree as ET

import numpy as np

from .util import read_image


class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`

    `VOC`: https://pjreddie.com/projects/pascal-voc-dataset-mirror/

    The index corresponds to each image.

    When queried by an index, if  `return_difficult == False`,
    this dataset returns a corresponding
     `img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If  `return_difficult == True`, this dataset returns corresponding
     `img, bbox, label, difficult`.  `difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    `(R, 4)`, where `R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are `(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape `(R,)`.
    `R` is the number of bounding boxes in the image.
    The class name of the label `l` is `l` th element of
     `VOC_BBOX_LABEL_NAMES`.

    The array  `difficult` is a one dimensional boolean array of shape
    `(R,)`. `R` is the number of bounding boxes in the image.
    If  `use_difficult` is  `False`, this array is
    a boolean array with all `False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * `img.dtype == numpy.float32`
    * `bbox.dtype == numpy.float32`
    * `label.dtype == numpy.int32`
    * `difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. `test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in `year`. TODO: add support for 2012
        use_difficult (bool): If `True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If `True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is `False`.

    """

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):
        '''
        attention: if it is 2007 dataset, it's OK to use the test as a splits.
        if it is 2012 dataset, since some images in test don't have corresponding annotations, the splits can only be
        train, trainval, val.
        '''
        id_list_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # skip the obect is difficult if using_difficult=false 
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')
