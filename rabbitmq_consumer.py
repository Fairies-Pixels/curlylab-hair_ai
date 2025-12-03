import pika
import json
import base64
import datetime
from sys import stderr

RABBIT_HOST = "rabbitmq"
REQUEST_QUEUE = "consistence.requests"
RESPONSE_QUEUE = "consistence.responses"
EXCHANGE = "consistence.exchange"

from consists_check_service import ocr_image_to_text,\
    analyze_text_composition,\
    INGREDIENT_DB

def process_consistence_request(ch, method, properties, body):
    try:
        print("Get request")
        request_data = json.loads(body)

        image_data = base64.b64decode(request_data['file'])

        raw_text = ocr_image_to_text(image_data)

        if not raw_text.strip():
            response = {
                "error": "После OCR не осталось текста для анализа."
            }
        else:
            issues = analyze_text_composition(raw_text, INGREDIENT_DB)

            pretty_issues = [
                {
                    "ingredient": issue["ingredient"],
                    "category": issue.get("category"),
                    "reason": issue.get("reason")
                }
                for issue in issues
            ]

            if not pretty_issues:
                response = {
                    "ok": True,
                    "raw_text_excerpt": raw_text[:200],
                    "issues_count": 0,
                    "issues": [],
                    "result": "Состав отличный!"
                }
            else:
                response = {
                    "ok": True,
                    "raw_text_excerpt": raw_text[:200],
                    "issues_count": len(pretty_issues),
                    "issues": pretty_issues,
                    "result": "Некоторые ингредиенты могут не подойти"
                }

        ch.basic_publish(
            exchange=EXCHANGE,
            routing_key='consistence.response.bind',
            body=json.dumps(response)
        )

        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"Processed consistence analysis request", file=stderr)

    except Exception as e:
        print(f"Error processing request: {e}", file=stderr)
        error_response = {
            "ok": False,
            "error": str(e)
        }
        ch.basic_publish(
            exchange=EXCHANGE,
            routing_key='consistence.response.bind',
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

        channel.queue_bind(exchange=EXCHANGE, queue=REQUEST_QUEUE, routing_key='consistence.request.bind')
        channel.queue_bind(exchange=EXCHANGE, queue=RESPONSE_QUEUE, routing_key='consistence.response.bind')
        print(f"[{datetime.datetime.now()}]: Queues and exchange binded...", file=stderr)

        channel.basic_consume(
            queue=REQUEST_QUEUE,
            on_message_callback=process_consistence_request
        )

        print("RabbitMQ consumer started for consistence analysis!", file=stderr)
        channel.start_consuming()

    except Exception as e:
        print(f"Failed to start RabbitMQ consumer: {e}", file=stderr)

if __name__ == "__main__":
    start_rabbitmq_consumer()