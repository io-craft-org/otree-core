from starlette.responses import JSONResponse

import otree
from otree.models import Participant
from .cbv import BaseRESTView


class RESTSessionParticipants(BaseRESTView):
    url_pattern = "/api/sessions/{code}/participants"

    # Output format:
    # [
    #     {
    #         "_current_app_name": "FGES_Louvain",
    #         "_current_page": "5/20",
    #         "_current_page_name": "Test4_Page",
    #         "_last_page_timestamp": 1634117391,
    #         "_monitor_note": null,
    #         "_numeric_label": "P1",
    #         "_round_number": 1,
    #         "code": "ud4myvih",
    #         "id_in_session": 1,
    #         "label": null
    #     },
    #     {
    #         "_current_app_name": "FGES_Louvain",
    #         "_current_page": "1/20",
    #         "_current_page_name": "ConsignesPage",
    #         "_last_page_timestamp": 1634117242,
    #         "_monitor_note": null,
    #         "_numeric_label": "P2",
    #         "_round_number": 1,
    #         "code": "1ebmc3rq",
    #         "id_in_session": 2,
    #         "label": null
    #     },
    #     {
    #         "_current_app_name": "FGES_Louvain",
    #         "_current_page": "10/20",
    #         "_current_page_name": "Matrices_2",
    #         "_last_page_timestamp": 1634117262,
    #         "_monitor_note": null,
    #         "_numeric_label": "P3",
    #         "_round_number": 1,
    #         "code": "2vhhf0iw",
    #         "id_in_session": 3,
    #         "label": null
    #     }
    # ]
    def get(self):
        code = self.request.path_params["code"]
        participants = Participant.objects_filter(_session_code=code, visited=True)
        data = otree.export.get_rows_for_monitor(participants)
        return JSONResponse(data)
