from io import StringIO

from starlette.responses import JSONResponse, Response

import otree
import otree.export
from otree.channels.utils import auto_advance_group
from otree.database import db
from otree.models import Participant
from .cbv import BaseRESTView
from .export import get_csv_http_response


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


def advance_participant(participant):
    page_index = participant._index_in_pages

    if page_index == 0:
        participant.initialize(None)
        participant._visit_current_page()
    else:
        participant._submit_current_page()
        participant._visit_current_page()

        otree.channels.utils.sync_group_send(
            group=auto_advance_group(participant.code), data={"auto_advanced": True}
        )


class AdvanceSessionParticipantView(BaseRESTView):
    url_pattern = "/api/participants/{code}/advance"

    def post(self):
        code = self.request.path_params["code"]
        participant = db.get_or_404(Participant, code=code)
        advance_participant(participant)
        return Response()


class ExportSession(BaseRESTView):

    url_pattern = "/api/sessions/{code}/export/app/{app_name}"

    def get(self):
        code = self.request.path_params["code"]
        app_name = self.request.path_params["app_name"]
        buf = StringIO()
        otree.export.export_app(app_name=app_name, fp=buf, session_code=code)
        return get_csv_http_response(buf, f"session_{code}_data")
