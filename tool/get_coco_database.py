import fiftyone.zoo as foz
import fiftyone as fo
from fiftyone import ViewField as F

classes=["person","car","cat","dog"]

dataset = foz.load_zoo_dataset(
    "coco-2017",
    splits=["train", "validation", "test"],
    label_types=["detections"],
    classes=classes,
    max_samples=200,
)

view = dataset.filter_labels("detections", F("label").is_in(classes))
dataset_type = fo.types.COCODetectionDataset

#session = fo.launch_app(view, remote=True)
print('done')

view.export(export_dir=r'C:\Users\kfir.gedalyahu\fiftyone\test2', dataset_type=dataset_type)


