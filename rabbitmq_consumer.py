import io

import pika
import json
import base64
import datetime
import torchvision.transforms as transforms
from sys import stderr

from PIL import Image

RABBIT_HOST = "rabbitmq"
REQUEST_QUEUE = "hairType.requests"
RESPONSE_QUEUE = "hairType.responses"
EXCHANGE = "hairType.exchange"

from models_loader import load_porosity_model
from main import predict_porosity

porosity_model = load_porosity_model()

porosity_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def process_porosity_request(ch, method, properties, body):
    try:
        print("Get request")

        request_data = json.loads(body)

        image_data = base64.b64decode(request_data['file'])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        porosity = predict_porosity(image)

        response = {
            "porosity": porosity
        }

        ch.basic_publish(
            exchange=EXCHANGE,
            routing_key='hairType.response.bind',
            body=json.dumps(response)
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"Processed porosity analysis request", file=stderr)

    except Exception as e:
        print(f"Error processing request: {e}", file=stderr)
        error_response = {
            "ok": False,
            "error": str(e)
        }
        ch.basic_publish(
            exchange=EXCHANGE,
            routing_key='hairType.response.bind',
            body=json.dumps(error_response)
        )

def start_rabbitmq_consumer():
    print("Starting RabbitMQ consumer...", file=stderr)
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBIT_HOST))
        print(f"[{datetime.datetime.now()}]: Connected to host", file=stderr)

        channel = connection.channel()
        print(f"[{datetime.datetime.now()}]: Channel selected...", file=stderr)

        channel.exchange_declare(exchange=EXCHANGE, durable=True, exchange_type='direct')
        channel.queue_declare(queue=REQUEST_QUEUE, durable=True)
        channel.queue_declare(queue=RESPONSE_QUEUE, durable=True)
        print(f"[{datetime.datetime.now()}] Queues and exchange set...", file=stderr)

        channel.queue_bind(exchange=EXCHANGE, queue=REQUEST_QUEUE, routing_key='hairType.request.bind')
        channel.queue_bind(exchange=EXCHANGE, queue=RESPONSE_QUEUE, routing_key='hairType.response.bind')
        print(f"[{datetime.datetime.now()}]: Queues and exchange binded...", file=stderr)

        channel.basic_consume(
            queue=REQUEST_QUEUE,
            on_message_callback=process_porosity_request
        )

        print("RabbitMQ consumer started for porosity analysis!", file=stderr)
        channel.start_consuming()

    except Exception as e:
        print(f"Failed to start RabbitMQ consumer: {e}", file=stderr)

if __name__ == "__main__":
    start_rabbitmq_consumer()