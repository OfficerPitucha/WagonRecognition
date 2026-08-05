"""
Cloud API wrapper for the train detection pipeline.

This module provides a thin integration layer between the local pipeline and the
cloud backend. The two core methods (:meth:`CloudAPI.send_start_message` and
:meth:`CloudAPI.send_end_message`) define *where* and *when* messages are sent;
the actual HTTP logic is intentionally left as stubs so the cloud team can fill
in the endpoint details, authentication, and error handling without touching
pipeline code.

Usage example::

    api = CloudAPI()
    api.send_start_message()          # call when a new train is detected
    result = pipeline.process_train(segment)
    api.send_end_message(result)      # call when the pipeline finishes
"""

import json


class CloudAPI:
    """Wrapper for all outgoing API calls to the cloud backend.

    Instantiate once and reuse across trains. Each method corresponds to one
    logical event in the pipeline lifecycle. Replace the ``print`` stubs with
    real HTTP calls (e.g. ``requests.post``) when the cloud endpoint is ready.
    """

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def send_start_message(self) -> None:
        """Notify the cloud that a new train has been detected and processing has started.

        This should be called as soon as motion is confirmed and the pipeline
        begins consuming a video segment. The message lets the backend open a
        new train record before results arrive.

        TODO: Replace the print stub with a POST request to the cloud endpoint,
              e.g. ``requests.post(ENDPOINT_START, json=payload)``.
        """
        payload = {"event": "train_processing_started"}
        print("[CloudAPI] send_start_message:", json.dumps(payload, indent=2))

    def send_end_message(self, json_object: dict) -> None:
        """Send the completed pipeline result for one train to the cloud.

        Called at the very end of the pipeline, after all wagons have been
        analysed and the result dict has been assembled. ``json_object`` is
        the same structure that is written to ``train_N_result.json``.

        Args:
            json_object: Full pipeline result for a single train (timestamp,
                         video name, total wagons, per-wagon details).

        TODO: Replace the print stub with a POST request to the cloud endpoint,
              e.g. ``requests.post(ENDPOINT_RESULT, json=json_object)``.
        """
        print("[CloudAPI] send_end_message:", json.dumps(json_object, indent=2))

    def send_end_message_open(self, json_object: dict) -> None:
        """Reserved for future use - alternative or extended end-of-train message.

        This method is provided as an extension point and is not called
        anywhere in the current pipeline. Implement and wire it up when the
        cloud team defines the corresponding endpoint.

        Args:
            json_object: Intended to carry pipeline result data (exact schema
                         TBD by the cloud team).
        """
        pass
