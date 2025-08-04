import json
import os
import random
from datetime import (
    datetime,
    timedelta
)
from typing import (
    List,
    Dict
)
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

'''

    This script generates synthetic travel data for a conversational travel assistant platform.
    It creates flight search data, visa rules, and airline refund policies using Gemini 2.5 Flash.
    The generated data is saved in JSON and Markdown formats for easy integration with the application.

'''

class TravelDataGenerator:

    def __init__(self):
        self.llm=ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            convert_system_message_to_human=True,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.airlines_data={
            "Star Alliance": [
                "United Airlines", "Lufthansa", "Air Canada", "Singapore Airlines",
                "Turkish Airlines", "Thai Airways", "All Nippon Airways", "Swiss International"
            ],
            "OneWorld": [
                "American Airlines", "British Airways", "Cathay Pacific", "Qatar Airways",
                "Japan Airlines", "Qantas", "Iberia", "Finnair"
            ],
            "SkyTeam": [
                "Delta Air Lines", "Air France", "KLM", "Korean Air",
                "China Eastern", "Alitalia", "Aeromexico", "Vietnam Airlines"
            ],
            "Non-Alliance": [
                "Emirates", "Etihad Airways", "JetBlue Airways", "Southwest Airlines",
                "Virgin Atlantic", "Norwegian Air", "Spirit Airlines", "Ryanair"
            ]
        }
        self.cities=[
            {"city": "Dubai", "country": "UAE", "airport": "DXB"},
            {"city": "Tokyo", "country": "Japan", "airport": "NRT"},
            {"city": "London", "country": "UK", "airport": "LHR"},
            {"city": "New York", "country": "USA", "airport": "JFK"},
            {"city": "Paris", "country": "France", "airport": "CDG"},
            {"city": "Singapore", "country": "Singapore", "airport": "SIN"},
            {"city": "Frankfurt", "country": "Germany", "airport": "FRA"},
            {"city": "Los Angeles", "country": "USA", "airport": "LAX"},
            {"city": "Sydney", "country": "Australia", "airport": "SYD"},
            {"city": "Hong Kong", "country": "Hong Kong", "airport": "HKG"},
            {"city": "Istanbul", "country": "Turkey", "airport": "IST"},
            {"city": "Amsterdam", "country": "Netherlands", "airport": "AMS"},
            {"city": "Bangkok", "country": "Thailand", "airport": "BKK"},
            {"city": "Seoul", "country": "South Korea", "airport": "ICN"},
            {"city": "Mumbai", "country": "India", "airport": "BOM"},
            {"city": "Cairo", "country": "Egypt", "airport": "CAI"},
            {"city": "Madrid", "country": "Spain", "airport": "MAD"},
            {"city": "Rome", "country": "Italy", "airport": "FCO"},
            {"city": "Toronto", "country": "Canada", "airport": "YYZ"},
            {"city": "Doha", "country": "Qatar", "airport": "DOH"}
        ]

    #   Generate synthetic flight data with realistic attributes.

    def generate_flight_data(self, num_flights: int=500) -> List[Dict]:
        flights=[]
        for _ in range(num_flights):
            from_city=random.choice(self.cities)
            to_city=random.choice([c for c in self.cities if c != from_city])
            alliance=random.choice(list(self.airlines_data.keys()))
            airline=random.choice(self.airlines_data[alliance])
            base_date=datetime.now()
            departure_date=base_date+timedelta(
                days=random.randint(1, 180),
                hours=random.randint(0, 23),
                minutes=random.choice([0, 15, 30, 45])
            )
            return_date=departure_date+timedelta(
                days=random.randint(1, 30),
                hours=random.randint(0, 23),
                minutes=random.choice([0, 15, 30, 45])
            )
            layovers=self._generate_layovers(from_city, to_city)
            
            base_price=random.randint(300, 2500)
            price_multiplier=1.0
            travel_class=random.choices(
                ["economy", "premium_economy", "business", "first"],
                weights=[60, 20, 15, 5]
            )[0]
            if travel_class=="premium_economy":
                price_multiplier=1.5
            elif travel_class=="business":
                price_multiplier=3.0
            elif travel_class=="first":
                price_multiplier=5.0
            price=int(base_price * price_multiplier)
            flight={
                "flight_id": f"FL{random.randint(1000, 9999)}",
                "airline": airline,
                "alliance": alliance,
                "from": from_city["city"],
                "from_airport": from_city["airport"],
                "from_country": from_city["country"],
                "to": to_city["city"],
                "to_airport": to_city["airport"],
                "to_country": to_city["country"],
                "departure_date": departure_date.isoformat(),
                "return_date": return_date.isoformat(),
                "travel_class": travel_class,
                "layovers": layovers,
                "layover_duration_hours": sum([l["duration_hours"] for l in layovers]),
                "price_usd": price,
                "refundable": random.choice([True, False]),
                "cancellation_fee_percent": random.choice([0, 5, 10, 15, 20]),
                "baggage_included": random.choice([True, False]),
                "wifi_available": random.choice([True, False]),
                "meal_service": random.choice(["none", "snack", "meal", "premium_meal"]),
                "flight_duration_hours": random.randint(2, 18),
                "aircraft_type": random.choice([
                    "Boeing 737", "Boeing 777", "Boeing 787", "Airbus A320",
                    "Airbus A330", "Airbus A350", "Airbus A380"
                ]),
                "availability": random.randint(0, 300)
            }
            flights.append(flight)
        return flights
    
    #   Generate layover information with realistic hub cities and durations.

    def _generate_layovers(self, from_city: Dict, to_city: Dict) -> List[Dict]:
        hub_cities=[
            {"city": "Dubai", "airport": "DXB"},
            {"city": "Istanbul", "airport": "IST"},
            {"city": "Doha", "airport": "DOH"},
            {"city": "Frankfurt", "airport": "FRA"},
            {"city": "Amsterdam", "airport": "AMS"},
            {"city": "London", "airport": "LHR"},
            {"city": "Singapore", "airport": "SIN"},
            {"city": "Hong Kong", "airport": "HKG"}
        ]
        num_layovers=random.choices([0, 1, 2], weights=[40, 50, 10])[0]
        if num_layovers==0:
            return []
        layovers=[]
        available_hubs=[h for h in hub_cities if h["city"] not in [from_city["city"], to_city["city"]]]
        for i in range(num_layovers):
            if not available_hubs:
                break
            hub=random.choice(available_hubs)
            available_hubs.remove(hub)
            layover={
                "city": hub["city"],
                "airport": hub["airport"],
                "duration_hours": random.choice([0.5, 1, 1.5, 2, 3, 4, 6, 12, 24])
            }
            layovers.append(layover)
        return layovers
    
    #   Generate visa rules and requirements using LLM.

    def generate_visa_rules(self) -> str:
        prompt=ChatPromptTemplate.from_messages([
            ("system", """You are a travel visa expert. Generate comprehensive visa rules and requirements 
            for international travel. Include information about:
             
            - Visa requirements for different passport holders
            - Duration of stay allowed
            - Documentation required
            - Processing times
            - Special conditions
            - Transit visa requirements
            
            Make the information realistic and detailed. Cover major countries and common travel scenarios."""),
            ("human", "Generate detailed visa rules and requirements for international travelers.")
        ])
        response=self.llm.invoke(prompt.format_messages())
        return response.content
    
    #   Generate airline refund and cancellation policies using LLM.

    def generate_refund_policies(self) -> str:
        prompt=ChatPromptTemplate.from_messages([
            ("system", """You are an airline policy expert. Generate comprehensive airline refund 
            and cancellation policies including:
             
            - Refundable vs non-refundable tickets
            - Cancellation timeframes and fees
            - Change fees and conditions  
            - Travel insurance recommendations
            - Special circumstances (medical, weather, etc.)
            - Different policies by ticket class
            
            Make the policies realistic and cover various scenarios travelers might encounter."""),
            ("human", "Generate detailed airline refund and cancellation policies.")
        ])
        response=self.llm.invoke(prompt.format_messages())
        return response.content
    
#   Driver function.

def main():
    generator=TravelDataGenerator()
    print("🚀 Generating synthetic travel data...")
    os.makedirs("data", exist_ok=True)
    print("✈️ Generating flight data...")
    flights=generator.generate_flight_data(500)
    with open(r"data/flights.json", "w") as f:
        json.dump(flights, f, indent=2)
    print(f"Generated {len(flights)} flight records")
    print("📋 Generating visa rules...")
    visa_rules=generator.generate_visa_rules()
    with open(r"data/visa_rules.md", "w") as f:
        f.write(visa_rules)
    print("Generated visa rules document")
    print("💰 Generating refund policies...")
    refund_policies=generator.generate_refund_policies()
    with open(r"data/refund_policies.md", "w") as f:
        f.write(refund_policies)
    print("Generated refund policies document")
    print("✅ Data generation complete!")
    print("\nGenerated files:")
    print("- data/flights.json (flight search data)")
    print("- data/visa_rules.md (visa requirements)")
    print("- data/refund_policies.md (airline policies)")

if __name__=="__main__":
    main()