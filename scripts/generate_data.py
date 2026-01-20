import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

fake= Faker()
Faker.seed(42)
random.seed(42)

def generate_clients(n_clients: int, output_path: str) -> list[int]:
    """
    Generate fake client data

    Args:
        n_clients (int): Numer of client to generate
        output_path (str): Path to save the client csv file

    Returns:
        list[int]: List of client IDs
    """

    countries = ["France", "Germany", "Spain", "Italy", "Belgium", 
                 "Netherland", "Switzerland", "UK", "Canada"]
    
    clients = []
    client_ids = []

    for i in range(1, n_clients + 1):
        date_inscription = fake.date_between(start_date = "-3y", end_date= "-1m")
        clients.append(
            {
                "id_client": i,
                "nom": fake.name(),
                "email": fake.email(),
                "date_inscription": date_inscription.strftime("%Y-%m-%d"),
                "pays": random.choice(countries)
            }
        )
        client_ids.append(i)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding= "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_client", "nom", "email", "date_inscription", "pays"])
        writer.writeheader()
        writer.writerows(clients)

    print(f"Generated Clients: {n_clients} in file {output_path} ")
    return client_ids

def generate_achats(client_ids: list[int], avg_purchases_per_client: int, output_path: str) -> None:
    """
    Generate fake purchase data.

    Args:
        client_ids: List of client IDs
        avg_purchases_per_client: Average number of purchases per client
        output_path: Path to save achats.csv
    """
    products = ["Laptop", "Phone", "Tablet", "Headphones", "Monitor", "Keyboard",
                "Mouse", "Webcam", "Speaker", "Charger"]

    achats = []
    achat_id = 1
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    for client_id in client_ids:
        n_purchases = random.randint(1, avg_purchases_per_client * 2)

        for _ in range(n_purchases):
            date_achat = fake.date_time_between(start_date=start_date, end_date=end_date)
            achats.append({
                "id_achat": achat_id,
                "id_client": client_id,
                "date_achat": date_achat.strftime("%Y-%m-%d %H:%M:%S"),
                "montant": round(random.uniform(10, 500), 2),
                "produit": random.choice(products)
            })
            achat_id += 1

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_achat", "id_client", "date_achat", "montant", "produit"])
        writer.writeheader()
        writer.writerows(achats)

    print(f"Generated {len(achats)} purchases -> {output_path}")



if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "data" / "sources"

    clients_ids = generate_clients(
        n_clients= 1500,
        output_path=str(output_dir / "clients.csv")
    )

    generate_achats(
        client_ids=clients_ids,
        avg_purchases_per_client=15,
        output_path=str(output_dir / "achats.csv")
    )


