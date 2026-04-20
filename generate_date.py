"""
============================================================
TRAINING DATA GENERATOR WITH CHECKPOINTING & RESUME
============================================================
Generates diverse training data with:
- 8 different instruction phrasings
- 5 question templates per context type
- Balanced distribution of question types
- High-quality extraction with validation
- Checkpoint saving for recovery from credit failures
- Resume capability for interrupted generation
============================================================
"""

import hashlib
import json
import math
import os
import random
import re
import time
import argparse
from collections import Counter
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import requests



API_KEY         = "sk-xxxxx"  
BASE_URL        = "https://api.deepseek.com/v1/chat/completions"
MODEL           = "deepseek-chat"

TEST_SAMPLES    = 10  # Small for testing
PRODUCTION_SAMPLES = 1000  # For final run
BATCH_SIZE      = 3
MAX_HEAL_ROUNDS = 3
OUTPUT_DIR      = "training_data"
CHECKPOINT_DIR  = "checkpoints"


INSTRUCTION_VARIANTS = {
    "formal": """प्रदान गरिएको सन्दर्भ मात्र प्रयोग गरी जवाफ दिनुहोस्।

नियमहरू:
1. केवल प्रश्नसँग प्रत्यक्ष सम्बन्धित जानकारी मात्र दिनुहोस्।
2. सन्दर्भमा नभएको कुनै पनि जानकारी नथप्नुहोस्।
3. सन्दर्भबाट आवश्यक वाक्यहरू मात्र निकाल्नुहोस्।
4. सम्पूर्ण सन्दर्भ नदोहोर्याउनुहोस्।
5. यदि उत्तर सन्दर्भमा छैन भने, ठ्याक्कै लेख्नुहोस्:
   "म यो जानकारी प्रदान गरिएको सन्दर्भमा पत्ता लगाउन सक्दिन।\"""",

    "concise": """दिइएको सन्दर्भ मात्र प्रयोग गर्दै उत्तर दिनुहोस्।
सन्दर्भ बाहिरको जानकारी नथप्नुहोस्।
उत्तर नभएमा: "म यो जानकारी प्रदान गरिएको सन्दर्भमा पत्ता लगाउन सक्दिन।\"""",

    "bullets": """नियमहरू:
• सन्दर्भबाट मात्र जवाफ दिनुहोस्
• नयाँ जानकारी नथप्नुहोस्
• सम्पूर्ण सन्दर्भ नदोहोर्याउनुहोस्
• उत्तर नभएमा: "म यो जानकारी प्रदान गरिएको सन्दर्भमा पत्ता लगाउन सक्दिन।\"""",

    "conversational": """कृपया तल दिइएको सन्दर्भ पढ्नुहोस् र प्रश्नको जवाफ दिनुहोस्।
ध्यान दिनुहोस्: सन्दर्भमा नभएको कुरा नलेख्नुहोस्।
यदि जवाफ छैन भने "म यो जानकारी प्रदान गरिएको सन्दर्भमा पत्ता लगाउन सक्दिन" भन्नुहोस्।""",

    "example_driven": """उदाहरण:
सन्दर्भ: "बचत खातामा ५% ब्याज छ"
प्रश्न: "ब्याजदर कति छ?"
उत्तर: "बचत खातामा ५% ब्याज छ"

अब यी नियमहरू पालना गर्दै उत्तर दिनुहोस्:
• सन्दर्भबाट मात्र जवाफ लेख्नुहोस्
• उत्तर नभएमा: "म यो जानकारी प्रदान गरिएको सन्दर्भमा पत्ता लगाउन सक्दिन।\"""",

    "strict_format": """FORMAT: Answer using ONLY context. NO external knowledge.
RULES:
1. Extract verbatim from context only
2. Do not add any information outside context
3. Do not copy entire context
4. If answer not in context → "म यो जानकारी प्रदान गरिएको सन्दर्भमा पत्ता लगाउन सक्दिन।\"""",

    "mixed_language": """Use केवल the provided context to answer.
Context बाहिरको information नहाल्नुहोस्।
If answer छैन, लेख्नुहोस्: "म यो जानकारी प्रदान गरिएको सन्दर्भमा पत्ता लगाउन सक्दिन।\"""",

    "very_short": """सन्दर्भ मात्र प्रयोग गर्नुहोस्।
उत्तर नभए: "म यो जानकारी प्रदान गरिएको सन्दर्भमा पत्ता लगाउन सक्दिन।\""""
}

# Distribution weights for instruction variants
INSTRUCTION_WEIGHTS = {
    "formal": 0.20,
    "concise": 0.20,
    "bullets": 0.15,
    "conversational": 0.15,
    "example_driven": 0.10,
    "strict_format": 0.10,
    "mixed_language": 0.05,
    "very_short": 0.05,
}

# ============================================================
# QUESTION TEMPLATES (Varying by context type & question type)
# ============================================================

QUESTION_TEMPLATES = {
    # Answerable question templates
    "answerable": {
        "numerical": [
            "{} कति हो?",
            "{} को दर कति छ?",
            "{} रकम कति तोकिएको छ?",
            "{} मा कति {} पाइन्छ?",
        ],
        "procedural": [
            "{} गर्न के के चाहिन्छ?",
            "{} को लागि कुन कागजात आवश्यक पर्छ?",
            "{} कसरी गर्न सकिन्छ?",
            "{} सक्रिय गर्न के गर्नुपर्छ?",
        ],
        "conditional": [
            "{} भएमा के हुन्छ?",
            "{} पुरा नगरेमा के परिणाम हुन्छ?",
            "{} को सर्त के के हुन्?",
            "कति रकम सम्म {} बिना {} गर्न सकिन्छ?",
        ],
        "comparative": [
            "{} र {} मा के फरक छ?",
            "{} को {} कति र {} को {} कति हो?",
            "{} भन्दा {} मा के बढी छ?",
        ]
    },
    
    "unavailable": {
        "numerical": [
            "{} को अधिकतम रकम कति हो?",
            "{} को न्यूनतम ब्याजदर कति हो?",
            "{} को शुल्क कति लाग्छ?",
            "{} कहिले सुरु भयो?",
            "{} को स्थापना मिति कति हो?",
            "{} को विदेशी मुद्रा दर कति हो?",
        ],
        "procedural": [
            "{} अनलाइनबाट गर्न सकिन्छ?",
            "{} को लागि कति दिन लाग्छ?",
            "{} गर्न कुन बैंक जानुपर्छ?",
        ],
        "conditional": [
            "{} भन्दा कम भएमा के हुन्छ?",
            "{} नगरेमा कुनै छुट पाइन्छ?",
            "{} को नियम कहिले परिवर्तन भयो?",
        ]
    },
    

    "multi_part": {
        "numerical": [
            "{} को {} र {} को {} कति कति हो?",
            "{} र {} को {} बीच के के फरक छन्?",
            "{} को {} के हो र {} को {} के हो?",
        ],
        "conditional": [
            "{} भन्दा बढी भएमा के गर्नुपर्छ र {} भन्दा कम भएमा के हुन्छ?",
            "{} गर्न के गर्नुपर्छ र {} नगरेमा के हुन्छ?",
            "कति रकम सम्म {} बिना {} गर्न सकिन्छ र कति भन्दा बढीमा {} चाहिन्छ?",
        ],
        "comparative": [
            "{} को {} र {} को {} के के हुन् र कुन राम्रो छ?",
            "{} र {} को {} को तुलना गर्नुहोस्।",
            "{} को {} र {} को {} बताउनुहोस् र फरक स्पष्ट गर्नुहोस्।",
        ]
    },
  
    "short_learning": {
        "definition": [
            "{} भनेको के हो?",
            "{} को परिभाषा के हो?",
            "{} लाई कसरी बुझ्नुहुन्छ?",
            "{} भन्नाले के बुझिन्छ?",
        ]
    },
}

SEED_CONTEXTS = [
    {
        "topic": "SAVINGS_ACCOUNT",
        "type": "numerical",
        "context": "बचत खाता खोल्नको लागि न्यूनतम १,००० रुपैयाँ जम्मा गर्नुपर्छ। खाता खोल्दा नागरिकता र पासपोर्ट साइज फोटो चाहिन्छ। बचत खातामा वार्षिक ५.५% ब्याज दिइन्छ।",
        "entities": {
            "thing": "बचत खाता",
            "amount": "१,००० रुपैयाँ",
            "rate": "५.५%",
        }
    },
    {
        "topic": "FIXED_DEPOSIT",
        "type": "numerical",
        "context": "मुद्दती निक्षेपमा न्यूनतम ५०,००० रुपैयाँ जम्मा गर्नुपर्छ। १ वर्षको लागि वार्षिक ७.५% ब्याज पाइन्छ। समय भन्दा पहिले निकाल्दा ०.५% जरिवाना कट्टी हुन्छ।",
        "entities": {
            "thing": "मुद्दती निक्षेप",
            "min_amount": "५०,००० रुपैयाँ",
            "rate": "७.५%",
            "penalty": "०.५%"
        }
    },
    {
        "topic": "HOME_LOAN",
        "type": "numerical",
        "context": "गृह कर्जाको अधिकतम अवधि २० वर्ष छ। न्यूनतम कर्जा रकम ५,००,००० रुपैयाँ हो। ब्याजदर वार्षिक ११% देखि १३% सम्म छ।",
        "entities": {
            "thing": "गृह कर्जा",
            "max_term": "२० वर्ष",
            "min_amount": "५,००,००० रुपैयाँ",
        }
    },
    {
        "topic": "MOBILE_BANKING",
        "type": "procedural",
        "context": "मोबाइल बैंकिङ सक्रिय गर्न बैंकमा दर्ता भएको मोबाइल नम्बर चाहिन्छ। सेवा सक्रिय भएपछि २४ घण्टामा प्रयोग गर्न सकिन्छ। मोबाइल बैंकिङ पिन ४ अंकको हुनुपर्छ।",
        "entities": {
            "thing": "मोबाइल बैंकिङ",
            "requirement": "बैंकमा दर्ता भएको मोबाइल नम्बर",
        }
    },
    {
        "topic": "KYC_UPDATE",
        "type": "procedural",
        "context": "KYC अपडेट गर्न नागरिकता, पासपोर्ट साइज फोटो र हालको ठेगाना प्रमाण चाहिन्छ। KYC प्रत्येक २ वर्षमा नवीकरण गर्नुपर्छ।",
        "entities": {
            "thing": "KYC",
            "documents": "नागरिकता, पासपोर्ट साइज फोटो, ठेगाना प्रमाण",
            "renewal_frequency": "२ वर्ष",
        }
    },
    {
        "topic": "REMITTANCE",
        "type": "conditional",
        "context": "विदेशबाट प्राप्त रेमिट्यान्स ५ लाख रुपैयाँ भन्दा बढी भएमा स्रोत प्रमाण पेश गर्नुपर्छ। ५ लाख सम्मको रेमिट्यान्स बिना कागजात प्राप्त गर्न सकिन्छ।",
        "entities": {
            "thing": "रेमिट्यान्स",
            "threshold": "५ लाख रुपैयाँ",
            "above_requirement": "स्रोत प्रमाण",
            "below_requirement": "बिना कागजात",
        }
    },
    {
        "topic": "CREDIT_SCORE",
        "type": "short_learning",
        "context": "क्रेडिट स्कोर भनेको व्यक्तिको ऋण तिर्ने क्षमता र इतिहासको आधारमा दिइने संख्यात्मक मूल्याङ्कन हो।",
        "entities": {
            "term": "क्रेडिट स्कोर",
            "definition": "व्यक्तिको ऋण तिर्ने क्षमता र इतिहासको आधारमा दिइने संख्यात्मक मूल्याङ्कन"
        }
    },
    {
        "topic": "DIGITAL_WALLET",
        "type": "procedural",
        "context": "डिजिटल वालेट खोल्न मोबाइल नम्बर र ईमेल ठेगाना चाहिन्छ। वालेटमा अधिकतम ५०,००० रुपैयाँ सम्म राख्न सकिन्छ।",
        "entities": {
            "thing": "डिजिटल वालेट",
            "requirements": "मोबाइल नम्बर, ईमेल ठेगाना",
            "max_balance": "५०,००० रुपैयाँ"
        }
    },
    {
        "topic": "BUSINESS_LOAN",
        "type": "conditional",
        "context": "व्यवसायिक कर्जाको लागि न्यूनतम २ वर्षको व्यवसाय दर्ता प्रमाण पत्र चाहिन्छ। १० लाख भन्दा कमको कर्जाको लागि धितो आवश्यक पर्दैन।",
        "entities": {
            "thing": "व्यवसायिक कर्जा",
            "requirement": "२ वर्षको व्यवसाय दर्ता प्रमाण पत्र",
            "threshold": "१० लाख",
            "condition": "धितो आवश्यक पर्दैन"
        }
    }
]

UNANSWERABLE = "म यो जानकारी प्रदान गरिएको सन्दर्भमा पत्ता लगाउन सक्दिन।"

def validate_sample(sample: dict) -> list[str]:
    """Validate sample against all rules"""
    issues = []
    output = sample.get("output", "")
    qtype = sample.get("metadata", {}).get("question_type", "")
    
    if qtype == "unavailable" and output.strip() != UNANSWERABLE:
        issues.append(f"unavailable type must use exact unanswerable string")
    
    if not output or len(output.strip()) < 5:
        issues.append("output too short or empty")
    
    return issues

def validate_multi_part(sample: dict) -> list[str]:
    """Ensure multi_part answers have required components"""
    issues = []
    if sample.get("metadata", {}).get("question_type") != "multi_part":
        return issues
    
    output = sample.get("output", "")
    question = sample.get("input", "")
    
    # Skip if output is unanswerable (partial answer case)
    if output.strip() == UNANSWERABLE:
        return issues
    
    # Count how many facts were asked
    if "र" in question or "?" in question:
        # Count question parts (sentences separated by र or ?)
        parts = question.count("र") + 1  # Simple count
        if "?" in question:
            parts = max(parts, 2)
        
        # Count sentences in output (। indicates sentence end)
        output_sentences = output.count("।") + 1 if output.count("।") > 0 else 1
        
        if parts > output_sentences:
            issues.append(f"Multi_part question has {parts} parts but output has only {output_sentences} sentence(s)")
    
    return issues

def validate_type_consistency(sample: dict) -> list[str]:
    """Ensure question_type matches output pattern"""
    issues = []
    qtype = sample.get("metadata", {}).get("question_type", "")
    output = sample.get("output", "")
    is_unanswerable = output.strip() == UNANSWERABLE
    
    if is_unanswerable and qtype not in ["unavailable", "multi_part"]:
        issues.append(f"Question type '{qtype}' but output is unanswerable. Should be 'unavailable'")
    
    if not is_unanswerable and qtype == "unavailable":
        issues.append(f"Question type 'unavailable' but output has answer: '{output[:50]}...'")
    
    return issues

def generate_question(context_data: Dict, question_type: str) -> str:
    """Generate varied question based on context type and question type"""
    
    context_type = context_data["type"]
    entities = context_data.get("entities", {})
    topic = context_data["topic"]
    
    # Get appropriate templates
    if question_type == "short_learning":
        templates = QUESTION_TEMPLATES["short_learning"]["definition"]
    else:
        templates = QUESTION_TEMPLATES.get(question_type, {}).get(context_type, ["{} को बारेमा प्रश्न?"])
    
    if not templates:
        templates = ["{} के हो?"]
    
    template = random.choice(templates)
    
    # Fill template with appropriate entities
    if question_type == "answerable":
        if context_type == "numerical":
            thing = entities.get("thing", topic)
            if "दर" in template or "ब्याज" in template:
                return template.format(f"{thing} को ब्याजदर")
            elif "रकम" in template:
                return template.format(f"{thing} को न्यूनतम रकम")
            else:
                return template.format(thing)
        
        elif context_type == "procedural":
            thing = entities.get("thing", topic)
            if "कागजात" in template:
                return template.format(f"{thing} अपडेट")
            else:
                return template.format(f"{thing} सक्रिय")
        
        elif context_type == "conditional":
            thing = entities.get("thing", topic)
            threshold = entities.get("threshold", "निश्चित रकम")
            if "भन्दा बढी" in template:
                return template.format(threshold)
            else:
                return template.format(thing)
        
        elif context_type == "comparative":
            thing1 = entities.get("thing1", "पहिलो")
            thing2 = entities.get("thing2", "दोस्रो")
            return template.format(thing1, thing2)
    
    elif question_type == "unavailable":
        thing = entities.get("thing", topic)
        return template.format(thing)
    
    elif question_type == "multi_part":
        if context_type == "numerical":
            thing = entities.get("thing", topic)
            attr1 = entities.get("rate", "दर")
            attr2 = entities.get("penalty", "जरिवाना")
            return template.format(thing, attr1, thing, attr2)
        
        elif context_type == "conditional":
            thing = entities.get("thing", topic)
            threshold = entities.get("threshold", "सीमा")
            requirement = entities.get("above_requirement", "कागजात")
            return template.format(threshold, requirement, thing)
        
        elif context_type == "comparative":
            return template.format("बचत खाता", "ब्याजदर", "चल्ती खाता", "ब्याजदर")
    
    elif question_type == "short_learning":
        term = entities.get("term", topic)
        return template.format(term)
    
    # Fallback
    return f"{topic} को बारेमा प्रश्न?"


_TYPE_CYCLE = [
    "answerable", "answerable", "unavailable", "answerable",
    "multi_part", "unavailable", "answerable", "short_learning",
    "multi_part", "unavailable"
]

def _type_schedule(n: int) -> list[str]:
    """Return a list of n question_types following balanced distribution"""
    cycle = _TYPE_CYCLE * math.ceil(n / len(_TYPE_CYCLE))
    return cycle[:n]


def select_context(question_type: str) -> Dict:
    """Select appropriate context for question type"""
    if question_type == "short_learning":
        preferred = [s for s in SEED_CONTEXTS if s["type"] == "short_learning"]
    elif question_type == "multi_part":
        preferred = [s for s in SEED_CONTEXTS if s["type"] in ["conditional", "comparative"]]
    elif question_type == "unavailable":
        preferred = [s for s in SEED_CONTEXTS if s["type"] in ["numerical", "procedural"]]
    else:
        preferred = SEED_CONTEXTS
    
    pool = preferred if preferred else SEED_CONTEXTS
    return random.choice(pool)

def call_api(prompt: str, temperature: float = 0.3) -> Tuple[Optional[str], Optional[Dict]]:
    """Call DeepSeek API with timeout and error handling. Returns (response, usage_info)"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a Nepali banking QA dataset generator. Follow extraction rules strictly."},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=90)
        resp.raise_for_status()
        response_json = resp.json()
        usage = response_json.get("usage", {})
        return response_json["choices"][0]["message"]["content"], usage
    except requests.exceptions.Timeout:
        print(f"  ❌ API timeout")
        return None, None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print(f"  ❌ Authentication failed - Check your API key")
        elif e.response.status_code == 429:
            print(f"  ❌ Rate limit exceeded - Slow down generation")
        else:
            print(f"  ❌ API HTTP error: {e}")
        return None, None
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return None, None

def call_api_with_retry(prompt: str, max_retries: int = 3, temperature: float = 0.3) -> Tuple[Optional[str], Optional[Dict]]:
    """Call API with retry logic. Returns (response, usage_info)"""
    for attempt in range(max_retries):
        response, usage = call_api(prompt, temperature)
        if response:
            return response, usage
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"    Retry {attempt + 1}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)
    return None, None

class CheckpointManager:
    """Manages saving/loading progress to resume after failures"""
    
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, samples: List[Dict], completed_count: int, 
                        type_schedule: List[str], instruction_list: List[str],
                        total_target: int):
        """Save current progress"""
        checkpoint = {
            "completed_count": completed_count,
            "samples": samples,
            "type_schedule": type_schedule,
            "instruction_list": instruction_list,
            "total_target": total_target,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_file = f"{self.checkpoint_dir}/checkpoint_{completed_count}.json"
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        # Also save current samples as backup
        backup_file = f"{self.checkpoint_dir}/samples_backup_{completed_count}.json"
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        print(f"  💾 Checkpoint saved: {completed_count} samples")
        
        # Remove old checkpoints (keep last 5)
        self._cleanup_old_checkpoints()
        
    def load_latest_checkpoint(self):
        """Load the most recent checkpoint"""
        checkpoints = sorted(Path(self.checkpoint_dir).glob("checkpoint_*.json"))
        if not checkpoints:
            return None, 0, None, None, 0
        
        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        with open(latest, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        
        print(f"  🔄 Resuming from checkpoint: {checkpoint['completed_count']} samples")
        return (checkpoint["samples"], 
                checkpoint["completed_count"],
                checkpoint["type_schedule"],
                checkpoint["instruction_list"],
                checkpoint["total_target"])
    
    def _cleanup_old_checkpoints(self):
        """Keep only last 5 checkpoints"""
        checkpoints = sorted(Path(self.checkpoint_dir).glob("checkpoint_*.json"))
        for old in checkpoints[:-5]:
            old.unlink()

# ============================================================
# CREDIT TRACKER
# ============================================================

class CreditTracker:
    """Tracks API usage and handles credit limits"""
    
    # DeepSeek pricing (as of 2024)
    # Update these based on your actual pricing
    COST_PER_1M_INPUT_TOKENS = 0.14   # USD
    COST_PER_1M_OUTPUT_TOKENS = 0.28  # USD
    
    def __init__(self, max_budget_usd: float = 10.0):
        self.max_budget_usd = max_budget_usd
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.credit_limit_reached = False
        
    def track_usage(self, usage: Dict):
        """Track token usage from API response"""
        if usage:
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            
            # Calculate cost
            input_cost = (input_tokens / 1_000_000) * self.COST_PER_1M_INPUT_TOKENS
            output_cost = (output_tokens / 1_000_000) * self.COST_PER_1M_OUTPUT_TOKENS
            self.total_cost += input_cost + output_cost
            
            # Check if approaching limit
            if self.total_cost >= self.max_budget_usd * 0.95:
                print(f"\n  ⚠️  WARNING: Used ${self.total_cost:.2f}/${self.max_budget_usd:.2f} ({(self.total_cost/self.max_budget_usd)*100:.1f}%)")
                if self.total_cost >= self.max_budget_usd:
                    self.credit_limit_reached = True
                    print(f"  ❌ Credit limit reached! Stopping generation.")
                    return False
        return True
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget in USD"""
        return max(0, self.max_budget_usd - self.total_cost)
    
    def estimate_samples_remaining(self, avg_cost_per_sample: float = 0.0003) -> int:
        """Estimate how many more samples can be generated"""
        remaining = self.get_remaining_budget()
        return int(remaining / avg_cost_per_sample)
    
    def print_summary(self):
        """Print usage summary"""
        print(f"\n  📊 Credit Usage Summary:")
        print(f"    Total cost: ${self.total_cost:.4f}")
        print(f"    Budget: ${self.max_budget_usd:.2f}")
        print(f"    Remaining: ${self.get_remaining_budget():.4f}")
        print(f"    Input tokens: {self.total_input_tokens:,}")
        print(f"    Output tokens: {self.total_output_tokens:,}")
        print(f"    Total tokens: {self.total_input_tokens + self.total_output_tokens:,}")


def generate_single_sample(question_type: str, instruction_variant: str, 
                          credit_tracker: Optional[CreditTracker] = None) -> Optional[Dict]:
    """Generate one QA sample with specified type and instruction"""
    
    # Select context
    context_data = select_context(question_type)
    context = context_data["context"]
    
    # Generate question
    question = generate_question(context_data, question_type)
    
    # Get instruction
    instruction = INSTRUCTION_VARIANTS[instruction_variant]
    
    # Prepare prompt for model
    prompt = f"""Generate a QA sample with these specifications:

Instruction: {instruction}

Context: {context}

Question: {question}

Question Type: {question_type}

Rules:
- If answerable: Extract exact answer from context verbatim
- If unavailable: Output exactly: "{UNANSWERABLE}"
- If multi_part: Extract ALL relevant sentences
- If short_learning: Extract only the definition sentence

Return ONLY this JSON object:
{{
    "instruction": "{instruction}",
    "input": "सन्दर्भ: {context}\\nप्रश्न: {question}",
    "output": "<answer here>",
    "metadata": {{
        "topic": "{context_data['topic']}",
        "question_type": "{question_type}",
        "difficulty": "medium",
        "context_type": "{context_data['type']}",
        "instruction_variant": "{instruction_variant}"
    }}
}}"""
    
    response, usage = call_api_with_retry(prompt, temperature=0.2)
    if not response:
        return None
    
    # Track usage if credit_tracker provided
    if credit_tracker and usage:
        credit_tracker.track_usage(usage)
    
    # Parse JSON
    try:
        # Clean response
        response = re.sub(r"```json\s*", "", response)
        response = re.sub(r"```\s*$", "", response)
        sample = json.loads(response.strip())
        
        if sample:
            # Run all validations
            issues = []
            issues.extend(validate_sample(sample))
            issues.extend(validate_multi_part(sample))
            issues.extend(validate_type_consistency(sample))
            
            if issues:
                print(f"    ⚠️ Validation issues: {issues}")
                return None
            return sample
    except json.JSONDecodeError as e:
        print(f"  Failed to parse JSON: {e}")
        print(f"  Response: {response[:200]}")
        return None
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return None

def generate_training_data(
    target_samples: int,
    resume: bool = False,
    max_budget_usd: float = 10.0,
    save_interval: int = 100
) -> Tuple[List[Dict], CreditTracker]:
    """Generate training data with checkpointing and credit monitoring"""
    
    print(f"\n{'='*60}")
    print(f"  Generating {target_samples} training samples")
    print(f"  Budget: ${max_budget_usd:.2f}")
    print(f"  Resume mode: {resume}")
    print(f"{'='*60}\n")
    
    # Initialize components
    checkpoint_manager = CheckpointManager()
    credit_tracker = CreditTracker(max_budget_usd=max_budget_usd)
    
    # Try to resume from checkpoint
    samples = []
    start_idx = 0
    type_schedule = []
    instruction_list = []
    
    if resume:
        samples, start_idx, type_schedule, instruction_list, saved_target = checkpoint_manager.load_latest_checkpoint()
        
        if samples is not None:
            print(f"  ✓ Loaded {len(samples)} existing samples")
            print(f"  ✓ Resuming from index {start_idx}")
            
            # Check if target changed
            if saved_target != target_samples:
                print(f"  ⚠️  Target changed from {saved_target} to {target_samples}")
                # Adjust schedules if needed
                if target_samples > saved_target:
                    # Need more samples, extend schedules
                    additional_needed = target_samples - len(type_schedule)
                    if additional_needed > 0:
                        additional_types = _type_schedule(additional_needed)
                        type_schedule.extend(additional_types)
                        
                        # Extend instruction list
                        current_instruction_count = len(instruction_list)
                        for variant, weight in INSTRUCTION_WEIGHTS.items():
                            count = int(target_samples * weight) - current_instruction_count
                            if count > 0:
                                instruction_list.extend([variant] * count)
                        while len(instruction_list) < target_samples:
                            instruction_list.append(random.choice(list(INSTRUCTION_VARIANTS.keys())))
                        random.shuffle(instruction_list)
        else:
            resume = False  # No checkpoint found, start fresh
    
    if not resume or samples is None:
        # Start fresh
        samples = []
        start_idx = 0
        
        # Create schedules
        type_schedule = _type_schedule(target_samples)
        instruction_list = []
        for variant, weight in INSTRUCTION_WEIGHTS.items():
            count = int(target_samples * weight)
            instruction_list.extend([variant] * count)
        
        # Fill remaining if any
        while len(instruction_list) < target_samples:
            instruction_list.append(random.choice(list(INSTRUCTION_VARIANTS.keys())))
        random.shuffle(instruction_list)
    
    # Generate remaining samples
    failed_count = 0
    last_checkpoint = start_idx
    
    for i in range(start_idx, target_samples):
        # Check if credit limit reached
        if credit_tracker.credit_limit_reached:
            print(f"\n  ❌ Credit limit reached at {len(samples)}/{target_samples} samples")
            break
        
        # Show progress periodically with cost info
        if i % 50 == 0 and i > 0:
            print(f"\n  📊 Progress: {i}/{target_samples} ({i/target_samples*100:.1f}%)")
            print(f"  💰 Cost so far: ${credit_tracker.total_cost:.4f}")
            remaining_estimate = credit_tracker.estimate_samples_remaining()
            if remaining_estimate > 0:
                print(f"  📈 Estimated remaining samples with budget: ~{remaining_estimate}")
        
        question_type = type_schedule[i]
        instruction_variant = instruction_list[i]
        
        print(f"  [{i+1}/{target_samples}] Generating {question_type} with {instruction_variant}...", end=" ", flush=True)
        
        sample = generate_single_sample(question_type, instruction_variant, credit_tracker)
        
        if sample:
            samples.append(sample)
            print(f"✓")
        else:
            failed_count += 1
            print(f"✗")
        
        # Save checkpoint periodically
        if len(samples) - last_checkpoint >= save_interval:
            checkpoint_manager.save_checkpoint(
                samples, len(samples), 
                type_schedule, instruction_list,
                target_samples
            )
            last_checkpoint = len(samples)
        
        # Small delay to avoid rate limiting
        time.sleep(0.3)
        
        # Stop if credit limit reached
        if credit_tracker.credit_limit_reached:
            # Save final checkpoint before stopping
            checkpoint_manager.save_checkpoint(
                samples, len(samples), 
                type_schedule, instruction_list,
                target_samples
            )
            break
    
    print(f"\n  ✓ Generated {len(samples)} valid samples ({failed_count} failed)")
    credit_tracker.print_summary()
    
    return samples, credit_tracker


def save_samples(samples: List[Dict], path: str):
    """Save samples to JSON file"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Saved {len(samples)} samples → {path}")

def save_as_jsonl(samples: List[Dict], path: str):
    """Save as JSONL for LoRA fine-tuning"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            # Convert to Alpaca format
            alpaca_format = {
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", ""),
                "output": sample.get("output", "")
            }
            f.write(json.dumps(alpaca_format, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(samples)} samples as JSONL → {path}")

def save_metadata(samples: List[Dict], credit_tracker: CreditTracker, path: str):
    """Save generation metadata"""
    metadata = {
        "total_samples": len(samples),
        "total_cost_usd": credit_tracker.total_cost,
        "total_input_tokens": credit_tracker.total_input_tokens,
        "total_output_tokens": credit_tracker.total_output_tokens,
        "budget_usd": credit_tracker.max_budget_usd,
        "generation_date": datetime.now().isoformat(),
        "question_type_distribution": dict(Counter(
            s.get("metadata", {}).get("question_type", "unknown") for s in samples
        )),
        "instruction_distribution": dict(Counter(
            s.get("metadata", {}).get("instruction_variant", "unknown") for s in samples
        ))
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Saved metadata → {path}")


def quality_report(samples: List[Dict]):
    """Generate quality report with enhanced validation"""
    print(f"\n{'='*60}")
    print("  QUALITY REPORT")
    print(f"{'='*60}")
    
    # Counters
    type_counts = Counter()
    instruction_counts = Counter()
    valid_count = 0
    all_issues = []
    
    for i, s in enumerate(samples):
        qtype = s.get("metadata", {}).get("question_type", "unknown")
        ivar = s.get("metadata", {}).get("instruction_variant", "unknown")
        type_counts[qtype] += 1
        instruction_counts[ivar] += 1
        
        # Run all validation functions
        issues = []
        issues.extend(validate_sample(s))
        issues.extend(validate_multi_part(s))
        issues.extend(validate_type_consistency(s))
        
        if not issues:
            valid_count += 1
        else:
            all_issues.append({
                "sample_index": i,
                "topic": s.get("metadata", {}).get("topic"),
                "qtype": qtype,
                "issues": issues
            })
    
    # Print summary
    print(f"\n  Total samples: {len(samples)}")
    print(f"  Valid samples: {valid_count} ({valid_count/len(samples)*100:.1f}%)")
    
    # Print detailed issues if any
    if all_issues:
        print(f"\n  ⚠️  Issues found in {len(all_issues)} samples:")
        for issue in all_issues[:5]:  # Show first 5
            print(f"    Sample {issue['sample_index']+1} ({issue['qtype']}):")
            for iss in issue['issues']:
                print(f"      - {iss}")
        if len(all_issues) > 5:
            print(f"    ... and {len(all_issues) - 5} more samples with issues")
    
    print(f"\n  Question Type Distribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"    {qtype:<15}: {count:>3} ({count/len(samples)*100:.1f}%)")
    
    print(f"\n  Instruction Variant Distribution:")
    for ivar, count in sorted(instruction_counts.items()):
        print(f"    {ivar:<15}: {count:>3} ({count/len(samples)*100:.1f}%)")
    
    return valid_count


def display_samples(samples: List[Dict], max_display: int = 5):
    """Display first few samples"""
    print(f"\n{'='*60}")
    print(f"  SAMPLE PREVIEW (first {min(max_display, len(samples))})")
    print(f"{'='*60}")
    
    for i, s in enumerate(samples[:max_display]):
        print(f"\n  ── Sample {i+1} ──")
        print(f"  Type: {s.get('metadata', {}).get('question_type')}")
        print(f"  Instruction variant: {s.get('metadata', {}).get('instruction_variant')}")
        print(f"  Input: {s.get('input', '')[:150]}...")
        print(f"  Output: {s.get('output', '')[:150]}...")


def test_single_generation():
    """Test a single generation to verify everything works"""
    print("\n" + "="*60)
    print("  RUNNING SINGLE TEST GENERATION")
    print("="*60)
    
    test_sample = generate_single_sample("answerable", "formal")
    if test_sample:
        print("\n  ✓ Test successful!")
        print("\n  Sample output:")
        print(json.dumps(test_sample, ensure_ascii=False, indent=2))
        return True
    else:
        print("\n  ✗ Test failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate Nepali banking QA training data with checkpointing"
    )
    parser.add_argument(
        "--samples", "-n", 
        type=int, 
        default=TEST_SAMPLES,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--resume", "-r", 
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--budget", "-b", 
        type=float, 
        default=10.0,
        help="Maximum budget in USD (default: 10.0)"
    )
    parser.add_argument(
        "--save-interval", "-s", 
        type=int, 
        default=100,
        help="Save checkpoint every N samples (default: 100)"
    )
    parser.add_argument(
        "--test", "-t", 
        action="store_true",
        help="Run test generation only"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("  TRAINING DATA GENERATOR WITH CHECKPOINTING")
    print(f"{'='*60}")
    
    # Check API key
    if "sk-xxxxx" in API_KEY:
        print("\n  ❌ Please replace API_KEY with your actual DeepSeek key")
        print("  You can also set it as environment variable: DEEPSEEK_API_KEY")
        print("\n  Example: export DEEPSEEK_API_KEY='sk-your-key-here'")
        return
    
    # Run test if requested
    if args.test:
        test_single_generation()
        return
    
    # Generate samples
    samples, credit_tracker = generate_training_data(
        target_samples=args.samples,
        resume=args.resume,
        max_budget_usd=args.budget,
        save_interval=args.save_interval
    )
    
    if not samples:
        print("\n  ❌ No samples generated")
        return
    

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = f"{OUTPUT_DIR}/training_data_{timestamp}_{len(samples)}_samples.json"
    save_samples(samples, json_path)
    
    jsonl_path = f"{OUTPUT_DIR}/training_data_{timestamp}_{len(samples)}_samples.jsonl"
    save_as_jsonl(samples, jsonl_path)

    metadata_path = f"{OUTPUT_DIR}/metadata_{timestamp}_{len(samples)}_samples.json"
    save_metadata(samples, credit_tracker, metadata_path)

    display_samples(samples, min(5, len(samples)))

    quality_report(samples)
    
    if len(samples) < args.samples:
        print(f"\n{'='*60}")
        print("  ⚠️  PARTIAL GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"\n  Generated {len(samples)}/{args.samples} samples ({len(samples)/args.samples*100:.1f}%)")
        print(f"  Budget exhausted or credit limit reached")
        print(f"\n  To resume generation when credits are available:")
        print(f"    python {os.path.basename(__file__)} --samples={args.samples} --resume --budget={args.budget}")
        print(f"\n  Current checkpoint saved in: {CHECKPOINT_DIR}/")
    else:
        print(f"\n{'='*60}")
        print("  COMPLETE ✅")
        print(f"{'='*60}")
        print(f"\n  Successfully generated all {len(samples)} samples!")
        print(f"  Total cost: ${credit_tracker.total_cost:.4f}")
    
    print(f"\n  Files saved:")
    print(f"    - {json_path}")
    print(f"    - {jsonl_path}")
    print(f"    - {metadata_path}")

if __name__ == "__main__":
    main()