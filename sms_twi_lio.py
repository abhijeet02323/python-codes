from twilio.rest import Client

# Your Twilio credentials
account_sid = "your account sid"
auth_token = "your auth token"

client = Client(account_sid, auth_token)

# Sending the SMS
message = client.messages.create(
    body="write your messege here",
    from_='twilio number',  # Replace with your Twilio number
    to='recipient twilio no'       # Replace with the recipient's phone number
)

print(f"Message sent! SID: {message.sid}")
