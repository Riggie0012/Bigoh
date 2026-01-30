# sending an sms
import os
import africastalking

AFRICASTALKING_USERNAME = os.getenv("AFRICASTALKING_USERNAME", "")
AFRICASTALKING_API_KEY = os.getenv("AFRICASTALKING_API_KEY", "")
AFRICASTALKING_SENDER_ID = os.getenv("AFRICASTALKING_SENDER_ID", "Bigoh")

if AFRICASTALKING_USERNAME and AFRICASTALKING_API_KEY:
    africastalking.initialize(
        username=AFRICASTALKING_USERNAME,
        api_key=AFRICASTALKING_API_KEY,
    )
    sms = africastalking.SMS
else:
    sms = None


def send_sms(phone, message):
    if sms is None:
        print("SMS not configured: missing AFRICASTALKING_USERNAME or AFRICASTALKING_API_KEY.")
        return
    recipients = [phone]
    sender = AFRICASTALKING_SENDER_ID
    try:
        response = sms.send(message, recipients, sender)
        print(response)
    except Exception as error:
        print("Error is ", error)
