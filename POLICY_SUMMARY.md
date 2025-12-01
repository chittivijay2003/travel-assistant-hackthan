# Company Travel Policy Summary

## Overview
The travel assistant uses a **RAG (Retrieval-Augmented Generation)** system to ensure all travel recommendations comply with company budget policies. The policy is stored in a PDF and queried using Qdrant vector database.

## Policy PDF Location
📄 `data/policies/company_travel_policy.pdf`

## Policy Categories Implemented

### 1. Flight Booking Policy
- **Domestic Flights**: Economy class only, max $500 per ticket
- **International Flights**: Business class allowed for 6+ hour flights, max $2,500
- **Booking Window**: Minimum 2 weeks in advance
- **Route Preference**: Direct flights preferred; connecting only if 20%+ cost savings
- **Preferred Airlines**: 
  - Domestic: Delta, United, American Airlines
  - International: Emirates, Lufthansa

### 2. Ground Transportation Policy
- **Airport Transfers**: Taxi/ride-share up to $75 per trip
- **Daily Cab Usage**: Maximum $100 per day
- **Rental Cars**: Compact/mid-size only, max $60 per day
- **Public Transport**: Encouraged when available and safe
- **Mileage Reimbursement**: $0.58 per mile for personal vehicles

### 3. Accommodation Policy
- **Hotel Budget**:
  - Major cities: Up to $200 per night
  - Other locations: Up to $150 per night
- **Preferred Chains**: Marriott, Hilton, Hyatt
- **Room Type**: Standard room only (no suite upgrades)
- **Extended Stays** (7+ nights): Serviced apartments up to $175 per night

### 4. Meal and Per Diem Allowances
- **Breakfast**: $15 per day
- **Lunch**: $25 per day
- **Dinner**: $40 per day
- **Total Daily Allowance**: $80
- **Business Meals**: Up to $150 per person (with approval)

### 5. Total Travel Budget Limits
- **Domestic Trips** (3 days): Up to $1,500 total (flights + accommodation + transport)
- **International Trips** (5 days): Up to $4,000 total (flights + accommodation + transport)
- **Conference Attendance**: Additional $500 for registration and expenses

### 6. Booking Guidelines
- All travel must be booked through approved channels
- Travel dates/times should minimize work impact
- **Weekend Stays**: Allowed if 20%+ airfare savings
- **Travel Insurance**: Recommended for international trips, covered up to $100

### 7. Additional Benefits
- **Airport Lounge Access**: For flights 4+ hours or international travel
- **Wi-Fi Costs**: Reimbursed for in-flight and hotel internet
- **Baggage Fees**: One checked bag for trips 3+ days
- **Travel Accessories**: Up to $50 reimbursement per trip

## How RAG System Works

### 1. PDF Ingestion Process
```python
# The system loads the PDF and splits it into chunks
from src.workflow import TravelAssistantGraph

graph = TravelAssistantGraph()
graph.ingest_policies()  # Loads PDF into Qdrant vector database
```

### 2. Policy Retrieval During Travel Planning
When generating travel plans, the system:
1. Takes user's travel request (destination, dates, etc.)
2. Queries the policy PDF using semantic search
3. Retrieves relevant budget limits and guidelines
4. Uses this context to ensure recommendations stay within budget

### 3. Example Query Flow
```
User Request: "Plan a 5-day trip to London with flights and cabs"

RAG Query: "travel budget cab flight policy London international 5 days"

Retrieved Policy Context:
- International flights: Business class allowed for 6+ hours, max $2,500
- Airport transfers: Up to $75 per trip
- Daily cab usage: Maximum $100 per day
- International trips (5 days): Total budget up to $4,000

Generated Plan: 
✅ Recommends flights within $2,500
✅ Suggests cabs within daily $100 limit
✅ Ensures total cost stays under $4,000
✅ Notes policy compliance in response
```

## Technical Implementation

### RAG Manager (`src/utils/rag_manager.py`)
- **Embeddings**: Google Generative AI Embeddings (models/embedding-001)
- **Vector Store**: Qdrant (local or cloud)
- **Text Splitting**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Retrieval**: Semantic similarity search (top 3 chunks)

### Travel Plan Node Integration
The `TravelPlanNode` automatically:
1. Queries the policy context based on user request
2. Includes policy information in the LLM prompt
3. Ensures all flight, cab, and hotel recommendations comply
4. Adds "Policy Compliance Notes" to the response

## Current Status

✅ **Policy PDF Created**: `company_travel_policy.pdf` (4.2 KB)  
⚠️ **Qdrant Status**: Currently disabled (connection refused)  
ℹ️ **Fallback Mode**: When Qdrant is unavailable, system shows warning

## To Enable Full RAG System

### Option 1: Start Local Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Option 2: Use Qdrant Cloud
Update `.env`:
```
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
```

### Then Ingest Policies
```python
from src.workflow import TravelAssistantGraph

graph = TravelAssistantGraph()
graph.ingest_policies()
```

## Policy Effective Date
January 1, 2025

## Contact
For policy questions: travel@company.com
