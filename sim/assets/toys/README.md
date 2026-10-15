# Stage 1 toys

Five objects from Google Scanned Objects (Google Research, CC-BY 4.0), taken in their MuJoCo form
from [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects)
@ `6ff8d27`. Only `model.obj` and `texture.png` are kept; `toys.json` holds the scale, mass and
size from the Stage 0 exit table (`configs/control/stage0_facts.json`).

| Folder | Role | Note |
|---|---|---|
| `SCHOOL_BUS` | target | yellow school bus |
| `Thomas_Friends_Woodan_Railway_Henry` | target | green engine Henry (the GSO name has the typo "Woodan") |
| `FOOD_BEVERAGE_SET__2` | target | red ketchup bottle, lying; piece 2 of `FOOD_BEVERAGE_SET`, split out by `claude-docs/stage0_toys_20261013/tools/split.py` |
| `My_First_Wiggle_Crocodile` | distractor | green, same colour as Henry |
| `Thomas_Friends_Wooden_Railway_Talking_Thomas_z7yi7UFHJRj` | distractor | blue, same shape as Henry |

Google Scanned Objects: Downs et al., "Google Scanned Objects: A High-Quality Dataset of 3D
Scanned Household Items", ICRA 2022. https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/
