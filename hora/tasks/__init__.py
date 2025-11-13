from hora.tasks.allegro_hand_hora import AllegroHandHora
from hora.tasks.allegro_hand_grasp import AllegroHandGrasp

from hora.tasks.test_hand_hora import TestHandHora
from hora.tasks.test_hand_grasp import TestHandGrasp

# Mappings from strings to environments
isaacgym_task_map = {
    "AllegroHandHora": AllegroHandHora,
    "AllegroHandGrasp": AllegroHandGrasp,
    "PublicAllegroHandHora": AllegroHandHora,
    "PublicAllegroHandGrasp": AllegroHandGrasp,
    "TestHandHora": TestHandHora,
    "TestHandGrasp": TestHandGrasp,
}
