"""
Where problem reports go. This is the only file you need to edit to switch reporting on.

FORM_ENDPOINT
    The Formspree form URL, "https://formspree.io/f/<form id>". The form id is public by design: it can only deliver
    mail to the address you verified on formspree.io, so it is safe to ship inside the app. While it still contains
    REPLACE_WITH the app skips the web request and opens the player's mail program with the report instead.

MAILTO_ADDRESS
    Where that mail-program fallback addresses the report. Set it to your own address; it is a placeholder here on
    purpose, and no address of yours is stored anywhere else in the code.

No password or SMTP credential belongs here, and none is ever needed.
"""

FORM_ENDPOINT = "https://formspree.io/f/REPLACE_WITH_YOUR_FORM_ID"
MAILTO_ADDRESS = "reports@example.invalid"

REQUEST_TIMEOUT_S = 10             # for the HTTPS request
MIN_SECONDS_BETWEEN_REPORTS = 60   # one report per minute per app run
MAILTO_BODY_LIMIT = 1800           # characters of body in the mailto: fallback (mail clients truncate long links)
