import random
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

entities = [(i, fake.first_name()) for i in range(0, 100)]

top_cities = [
    "Tokyo", "Jakarta", "Delhi", "Guangzhou", "Mumbai", "Manila", "Shanghai", "Seoul",
    "Cairo", "Mexico City", "Kolkata", "Sao Paulo", "New York", "Karachi", "Dhaka",
    "Beijing", "Lagos", "Buenos Aires", "Istanbul", "Chongqing", "Rio de Janeiro",
    "Tianjin", "Kinshasa", "Lahore", "Bangalore", "Paris", "Bogota", "Chennai",
    "Lima", "Bangkok", "Hyderabad", "London", "Tehran", "Chicago", "Chengdu",
    "Nanjing", "Wuhan", "Berlin", "Luanda", "Ahmedabad", "Kuala Lumpur",
    "Hong Kong", "Hanoi", "Shenzhen", "Riyadh", "Baghdad", "Santiago", "Surat",
    "Madrid", "Singapore", "Pune", "Houston", "Dallas", "Toronto", "Miami", "Philadelphia",
    "Atlanta", "Khartoum", "Barcelona", "Washington", "Yangon", "Alexandria", "Harbin",
    "Los Angeles", "Shenyang", "Qingdao", "Zhengzhou", "Melbourne", "Jinan", "Cape Town",
    "Monterrey", "Durban", "Casablanca", "Brasilia", "Kabul", "Nairobi", "Rome",
    "Ankara", "Kano", "Mashhad", "Medellin", "Porto Alegre", "Recife", "Jeddah",
    "Addis Ababa", "Caracas", "Stockholm", "Vienna", "Osaka", "Nagoya",
    "Fukuoka", "Sendai", "Sapporo", "Busan", "Daegu", "Daejeon", "Gwangju", "Ulsan",
    "Incheon", "Yokohama"
]
assert len(top_cities) == 100

attributes = [(100+i, city) for i, city in enumerate(top_cities)]

relations_list = [
    "lives in", "was born in", "works in", "travels to", "studied in",
    "married in", "visited", "loves", "moved to", "left"
]
relations = [(200+i, relations_list[i]) for i in range(10)]

id_map = pd.DataFrame(entities + attributes + relations, columns=["id", "name"])
id_map.to_csv("id_mapping.csv", index=False)
print("Saved id_mapping.csv with", len(id_map), "entries")

