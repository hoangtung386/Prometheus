"""PUMA class names and stable model indices."""

TISSUE_CLASSES = [
    "background",
    "tumor",
    "stroma",
    "epidermis",
    "necrosis",
    "blood_vessel",
]
NUCLEI_CLASSES = [
    "background",
    "tumor",
    "stroma",
    "endothelium",
    "histiocyte",
    "melanophage",
    "lymphocyte",
    "plasma_cell",
    "neutrophil",
    "apoptosis",
    "epithelium",
]
TISSUE_CLASS_TO_IDX = {name: index for index, name in enumerate(TISSUE_CLASSES)}
NUCLEI_CLASS_TO_IDX = {name: index for index, name in enumerate(NUCLEI_CLASSES)}
