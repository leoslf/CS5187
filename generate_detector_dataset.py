import os
import h5py
import pickle
import pandas as pd

from streetnumber.detection import *
from sklearn.neighbors import KDTree

class DigitStruct:
    def __init__(self, filename):
        self.f = h5py.File(filename, "r")

        self.digitStructName = self.f["digitStruct"]["name"]
        self.digitStructBbox = self.f["digitStruct"]["bbox"]

    def get_name(self, n):
        name = "".join([chr(c[0]) for c in self.f[self.digitStructName[n][0]].value])
        # print (name)
        return name

    def bbox_helper(self, attr):
        return [self.f[attr.value[j].item()].value[0][0] for j in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]

    def get_bbox(self, n):
        bbox = self.digitStructBbox[n].item()
        x, y, w, h = np.array(list(map(lambda key: self.bbox_helper(self.f[bbox][key]), ["left", "top", "width", "height"])))
        bbox = np.column_stack((y, y+h, x, x+w))
        # print (bbox)
        return bbox.tolist()

    def get_dict(self):
        length = len(self.digitStructName)
        names = map(self.get_name, range(length))
        bboxes = map(self.get_bbox, range(length))
        return dict(zip(names, bboxes))

def bboxes_opencv_order(bboxes):
    return bboxes[:, [2, 0]].astype("int"), bboxes[:, [3, 1]].astype("int")

def bbox_centers(bboxes_opencv):
    a, b = bboxes_opencv
    return (a + b) // 2


def bbox_radius(bboxes_opencv):
    a, b = bboxes_opencv
    return np.linalg.norm((b - a) / 2, axis = 1)

def labels_from_regions(regions, bbox_centers, radiuses):
    regions_centers = regions.bbox_centers
    regions_radius = regions.bbox_radius

    kdtree = KDTree(regions_centers)

    mask = np.zeros(len(regions_centers))
    for (center, radius) in zip(bbox_centers, radiuses):
        indices = kdtree.query_radius(center.reshape(1, -1), radius)[0]
        sub_mask = np.zeros(len(regions_centers), dtype=int)
        sub_mask[indices] = 1

        mask[sub_mask & (regions_radius < 1.2 * radius) & (regions_radius >= 0.8 * radius)] = 1

    return mask


if __name__ == "__main__":
    # (X = patches, Y = binary labels)
    for dataset_name in ["train", "test"]:
        # digitStruct = DigitStruct(os.path.join(dataset_name, "digitStruct.mat"))
        # descriptor = digitStruct.get_dict()
        descriptor_filename = os.path.join(dataset_name, "descriptor.pickle")
        # with open(descriptor_filename, "wb") as f:
        #     pickle.dump(descriptor, f)
        #     print ("saved %s" % descriptor_filename)

        with open(descriptor_filename, "rb") as f:
            descriptor = pickle.load(f)

        Xs = []
        Ys = []

        Xs_filenames = []

        for i, (filename, bboxes) in enumerate(descriptor.items(), 1):
            if i > len(descriptor) * 0.01:
                break

            img_filename = os.path.join(dataset_name, filename)
            img = cv2.imread(img_filename)
            regions = Regions((32, 32, 1), img)
            bboxes = np.array(bboxes.copy(), dtype = "int")
            bboxes_opencv = bboxes_opencv_order(bboxes)
            centers = bbox_centers(bboxes_opencv)
            radiuses = bbox_radius(bboxes_opencv)

            labels = labels_from_regions(regions, centers, radiuses)

            print ("[%5d / %5d] img: %s" % (i, len(descriptor), img_filename))

            negative_class_indices = np.where(labels == 0)[0]
            index_to_delete = np.random.choice(negative_class_indices, size = int(len(negative_class_indices) * 0.8), replace = False)

            labels = np.delete(labels, index_to_delete, axis = 0)
            patches = regions.patches.copy()
            patches = np.delete(patches, index_to_delete, axis = 0)

            # Xs.append(regions.patches.copy())
            for j, patch in enumerate(patches, 1):
                patch_filename = "%d_%d.png" % (i, j)
                cv2.imwrite(os.path.join("%s/patches" % dataset_name, patch_filename), patch)
                print ("(%5d, %5d): %s" % (i, j, patch_filename))
                Xs.append(patch_filename)

            Ys.append(labels)

        # Xs = np.row_stack(Xs)
        Ys = np.concatenate(Ys).astype("int")

        # print ("Xs.shape: %r, Ys.shape: %r" % (Xs.shape, Ys.shape))

        df = pd.DataFrame({
            "image_id": Xs,
            "label": Ys
        })
        df_filename = "%s_patches.csv" % dataset_name
        df.to_csv(df_filename)
        print ("saved %s" % df_filename)




        # dataset_filename = os.path.join(dataset_name, "dataset.pickle")
        # with open(dataset_filename, "wb") as f:
        #     pickle.dump((Xs, Ys), f)
        #     print ("saved %s" % dataset_filename)


            




