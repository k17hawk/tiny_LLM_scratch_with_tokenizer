import hashlib
import json
import math
import os
import random
import re
import time
import argparse
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import requests


API_KEY             = "sk-555"         
BASE_URL            = "https://api.deepseek.com/v1/chat/completions"
MODEL               = "deepseek-chat"

TEST_SAMPLES        = 10
PRODUCTION_SAMPLES  = 1000
BATCH_SIZE          = 3
MAX_HEAL_ROUNDS     = 3
OUTPUT_DIR          = "training_data_english"
CHECKPOINT_DIR      = "checkpoints_english"

# Thread-safe lock for shared state mutations
_lock = threading.Lock()


INSTRUCTION_VARIANTS = {
    "formal": (
        "प्रदान गरिएको सन्दर्भ मात्र प्रयोग गरी जवाफ दिनुहोस्।\n\n"
        "नियमहरू:\n"
        "1. केवल प्रश्नसँग प्रत्यक्ष सम्बन्धित जानकारी मात्र दिनुहोस्।\n"
        "2. सन्दर्भमा नभएको कुनै पनि जानकारी नथप्नुहोस्।\n"
        "3. सन्दर्भबाट आवश्यक वाक्यहरू मात्र निकाल्नुहोस्।\n"
        "4. सम्पूर्ण सन्दर्भ नदोहोर्याउनुहोस्।\n"
        "5. यदि उत्तर सन्दर्भमा छैन भने, ठ्याक्कै लेख्नुहोस्:\n"
        '   "I cannot find this information in the provided context."'
    ),
    "concise": (
        "दिइएको सन्दर्भ मात्र प्रयोग गर्दै उत्तर दिनुहोस्।\n"
        "सन्दर्भ बाहिरको जानकारी नथप्नुहोस्।\n"
        'उत्तर नभएमा: "I cannot find this information in the provided context."'
    ),
    "bullets": (
        "नियमहरू:\n"
        "• सन्दर्भबाट मात्र जवाफ दिनुहोस्\n"
        "• नयाँ जानकारी नथप्नुहोस्\n"
        "• सम्पूर्ण सन्दर्भ नदोहोर्याउनुहोस्\n"
        '• उत्तर नभएमा: "I cannot find this information in the provided context."'
    ),
    "conversational": (
        "कृपया तल दिइएको सन्दर्भ पढ्नुहोस् र प्रश्नको जवाफ दिनुहोस्।\n"
        "ध्यान दिनुहोस्: सन्दर्भमा नभएको कुरा नलेख्नुहोस्।\n"
        'यदि जवाफ छैन भने "I cannot find this information in the provided context" भन्नुहोस्।'
    ),
    "example_driven": (
        "उदाहरण:\n"
        'सन्दर्भ: "बचत खातामा ५% ब्याज छ"\n'
        'प्रश्न: "ब्याजदर कति छ?"\n'
        'उत्तर: "बचत खातामा ५% ब्याज छ"\n\n'
        "अब यी नियमहरू पालना गर्दै उत्तर दिनुहोस्:\n"
        "• सन्दर्भबाट मात्र जवाफ लेख्नुहोस्\n"
        '• उत्तर नभएमा: "I cannot find this information in the provided context."'
    ),
    "strict_format": (
        "FORMAT: Answer using ONLY context. NO external knowledge.\n"
        "RULES:\n"
        "1. Extract verbatim from context only\n"
        "2. Do not add any information outside context\n"
        "3. Do not copy entire context\n"
        '4. If answer not in context → "I cannot find this information in the provided context."'
    ),
    "mixed_language": (
        "Use केवल the provided context to answer.\n"
        "Context बाहिरको information नहाल्नुहोस् र सम्पूर्ण context नदोहोर्याउनुहोस्।\n"
        'If answer छैन, लेख्नुहोस्: "I cannot find this information in the provided context."'
    ),
    "very_short": (
        "सन्दर्भ मात्र प्रयोग गर्नुहोस्।\n"
        'उत्तर नभए: "I cannot find this information in the provided context."'
    )
}
INSTRUCTION_WEIGHTS = {
    "formal":           0.20,
    "concise":          0.20,
    "bullets":          0.15,
    "conversational":   0.15,
    "example_driven":   0.10,
    "strict_format":    0.10,
    "mixed_language":   0.05,
    "very_short":       0.05,
}

# ============================================================
# QUESTION TEMPLATES  (English)
# ============================================================

QUESTION_TEMPLATES = {
    "answerable": {
        "numerical": [
            "How much is {}?",
            "What is the rate of {}?",
            "How much amount is prescribed for {}?",
            "How much {} is available in {}?",       # 2 placeholders
        ],
        "procedural": [
            "What is required for {}?",
            "Which documents are needed for {}?",
            "How can {} be done?",
            "What needs to be done to activate {}?",
        ],
        "conditional": [
            "What happens if {}?",
            "What is the consequence if {} is not fulfilled?",
            "What are the conditions of {}?",
            "Up to what amount can {} be done without {}?",  # 2 placeholders
        ],
        "comparative": [
            "What is the difference between {} and {}?",
            "How much is the {} of {} and the {} of {}?",   # 4 placeholders
            "What is more in {} compared to {}?",           # 2 placeholders
        ]
    },
    "unavailable": {
        "numerical": [
            "What is the maximum amount for {}?",
            "What is the minimum interest rate for {}?",
            "How much is the fee for {}?",
            "When did {} start?",
            "What is the establishment date of {}?",
            "What is the foreign currency rate for {}?",
        ],
        "procedural": [
            "Can {} be done online?",
            "How many days does it take for {}?",
            "Which bank should one go to for {}?",
        ],
        "conditional": [
            "What happens if it is less than {}?",
            "Is there any discount if {} is not done?",
            "When did the rules of {} change?",
        ]
    },
    "multi_part": {
        "numerical": [
            "How much is the {} of {} and the {} of {}?",   # 4 placeholders
            "What are the differences between the {} of {} and {}?",   # 3 placeholders
            "What is the {} of {} and the {} of {}?",   # 4 placeholders
        ],
        "conditional": [
            "What should be done if it exceeds {} and what happens if it is less than {}?",  # 2 placeholders
            "What should be done for {} and what happens if {} is not done?",   # 2 placeholders
            "Up to what amount can {} be done without {} and above how much is {} required?",  # 3 placeholders
        ],
        "comparative": [
            "What are the {} of {} and the {} of {}, and which one is better?",  # 4 placeholders
            "Compare the {} of {} and {}.",
            "Tell me the {} of {} and the {} of {}, and clarify the differences.",  # 4 placeholders
        ]
    },
    "short_learning": {
        "definition": [
            "What is {}?",
            "What is the definition of {}?",
            "How do you understand {}?",
            "What does {} mean?",
        ]
    },
}

# ============================================================
# SEED CONTEXTS  (unchanged – Nepali)
# ============================================================

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

UNANSWERABLE = "I cannot find this information in the provided context."

# ============================================================
# VALIDATION
# ============================================================

def validate_sample(sample: dict) -> list:
    issues = []
    output = sample.get("output", "")
    qtype = sample.get("metadata", {}).get("question_type", "")
    if qtype == "unavailable" and output.strip() != UNANSWERABLE:
        issues.append("unavailable type must use exact unanswerable string")
    if not output or len(output.strip()) < 5:
        issues.append("output too short or empty")
    return issues


def validate_multi_part(sample: dict) -> list:
    issues = []
    if sample.get("metadata", {}).get("question_type") != "multi_part":
        return issues
    output = sample.get("output", "")
    question = sample.get("input", "")
    if output.strip() == UNANSWERABLE:
        return issues
    if len(output.strip()) < 50 and "?" in question and question.count("?") > 1:
        issues.append(f"Multi_part answer too short ({len(output)} chars) for complex question")
    return issues


def validate_type_consistency(sample: dict) -> list:
    issues = []
    qtype = sample.get("metadata", {}).get("question_type", "")
    output = sample.get("output", "")
    is_unanswerable = output.strip() == UNANSWERABLE
    if is_unanswerable and qtype not in ["unavailable", "multi_part"]:
        issues.append(f"Question type '{qtype}' but output is unanswerable. Should be 'unavailable'")
    if not is_unanswerable and qtype == "unavailable":
        issues.append(f"Question type 'unavailable' but output has answer: '{output[:50]}...'")
    return issues

# ============================================================
# QUESTION GENERATION  (adapted to English templates)
# ============================================================

def generate_question(context_data: Dict, question_type: str) -> str:
    context_type = context_data["type"]
    entities = context_data.get("entities", {})
    topic = context_data["topic"]

    if question_type == "short_learning":
        templates = QUESTION_TEMPLATES["short_learning"]["definition"]
    else:
        templates = QUESTION_TEMPLATES.get(question_type, {}).get(context_type, ["What is {}?"])

    if not templates:
        templates = ["What is {}?"]

    template = random.choice(templates)
    placeholder_count = template.count("{}")
    args = []

    if question_type == "answerable":
        if context_type == "numerical":
            thing = entities.get("thing", topic)
            if placeholder_count == 1:
                # For single placeholder, just use the thing (rate/amount are handled by template)
                args = [thing]
            elif placeholder_count == 2:
                # "How much {} is available in {}?" → (attribute, thing)
                args = [entities.get("rate", "interest rate"), thing]
            else:
                args = [thing] * placeholder_count
        elif context_type == "procedural":
            thing = entities.get("thing", topic)
            # All procedural answerable templates have 1 placeholder
            args = [thing]
        elif context_type == "conditional":
            thing = entities.get("thing", topic)
            threshold = entities.get("threshold", "a certain amount")
            if placeholder_count == 1:
                # "What happens if {}?" → threshold; otherwise thing
                if "if" in template.lower():
                    args = [threshold]
                else:
                    args = [thing]
            elif placeholder_count == 2:
                # "Up to what amount can {} be done without {}?" → (thing, above_requirement)
                args = [thing, entities.get("above_requirement", "documents")]
            else:
                args = [thing] * placeholder_count
        elif context_type == "comparative":
            thing1 = entities.get("thing1", "the first")
            thing2 = entities.get("thing2", "the second")
            if placeholder_count == 2:
                args = [thing1, thing2]
            elif placeholder_count == 4:
                # "How much is the {} of {} and the {} of {}?" → attr1, thing1, attr2, thing2
                attr1 = entities.get("rate", "interest rate")
                attr2 = entities.get("penalty", "penalty") if "penalty" in entities else entities.get("rate", "rate")
                args = [attr1, thing, attr2, thing]   # note: thing reused (same entity both sides)
            else:
                args = [thing1] * placeholder_count

    elif question_type == "unavailable":
        thing = entities.get("thing", topic)
        args = [thing] * placeholder_count

    elif question_type == "multi_part":
        if context_type == "numerical":
            thing = entities.get("thing", topic)
            attr1 = entities.get("rate", "rate")
            attr2 = entities.get("penalty", "penalty") if "penalty" in entities else entities.get("amount", "amount")
            if placeholder_count == 4:
                args = [attr1, thing, attr2, thing]   # corresponds to "the {} of {} and the {} of {}"
            elif placeholder_count == 3:
                args = [attr1, thing, thing]           # "differences between the {} of {} and {}"
            elif placeholder_count == 2:
                args = [thing, attr1]
            else:
                args = [thing] * placeholder_count
        elif context_type == "conditional":
            thing = entities.get("thing", topic)
            threshold = entities.get("threshold", "a limit")
            requirement = entities.get("above_requirement", "documents")
            if placeholder_count == 2:
                args = [threshold, thing]   # for "if it exceeds {} and what happens if it is less than {}"
            elif placeholder_count == 3:
                args = [thing, requirement, thing]  # "Up to what amount can {} be done without {} and above how much is {} required?"
            else:
                args = [thing] * placeholder_count
        elif context_type == "comparative":
            if placeholder_count == 4:
                args = ["interest rate", "first", "interest rate", "second"]
            elif placeholder_count == 3:
                args = ["interest rate", "first", "second"]
            elif placeholder_count == 2:
                args = ["first", "second"]
            else:
                args = ["first"] * placeholder_count

    elif question_type == "short_learning":
        term = entities.get("term", topic)
        args = [term] * placeholder_count

    if not args:
        args = [topic] * placeholder_count
    while len(args) < placeholder_count:
        args.append(topic)
    args = args[:placeholder_count]

    return template.format(*args)


_TYPE_CYCLE = [
    "answerable", "answerable", "unavailable", "answerable",
    "multi_part", "unavailable", "answerable", "short_learning",
    "multi_part", "unavailable"
]


def _type_schedule(n: int) -> list:
    cycle = _TYPE_CYCLE * math.ceil(n / len(_TYPE_CYCLE))
    return cycle[:n]


def select_context(question_type: str) -> Dict:
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

# ============================================================
# API  (system message now requests English answers)
# ============================================================

def call_api(prompt: str, temperature: float = 0.3) -> Tuple[Optional[str], Optional[Dict]]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "You are a Nepali banking QA dataset generator. Answer in English, extracting only from the provided Nepali context."},
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
            print(f"  ❌ Rate limit hit - backing off...")
            time.sleep(5)   # extra back-off on 429 inside worker
        else:
            print(f"  ❌ API HTTP error: {e}")
        return None, None
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return None, None


def call_api_with_retry(prompt: str, max_retries: int = 3, temperature: float = 0.3) -> Tuple[Optional[str], Optional[Dict]]:
    for attempt in range(max_retries):
        response, usage = call_api(prompt, temperature)
        if response:
            return response, usage
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt
            print(f"    Retry {attempt + 1}/{max_retries} in {wait_time}s...")
            time.sleep(wait_time)
    return None, None

# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, samples: List[Dict], completed_count: int,
                        type_schedule: List[str], instruction_list: List[str],
                        total_target: int):
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
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

            backup_file = f"{self.checkpoint_dir}/samples_backup_{completed_count}.json"
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

            print(f"  💾 Checkpoint saved: {completed_count} samples → {checkpoint_file}")
            self._cleanup_old_checkpoints()
        except Exception as e:
            print(f"  ⚠️ Failed to save checkpoint: {e}")

    def load_latest_checkpoint(self):
        try:
            checkpoints = list(Path(self.checkpoint_dir).glob("checkpoint_*.json"))
            backups     = list(Path(self.checkpoint_dir).glob("samples_backup_*.json"))

            print(f"  🔍 Found {len(checkpoints)} checkpoints, {len(backups)} backups")

            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime) if checkpoints else None
            latest_backup     = max(backups, key=lambda x: x.stat().st_mtime) if backups else None

            def extract_count(path):
                return int(re.search(r'_(\d+)\.json$', path.name).group(1))

            checkpoint_count = extract_count(latest_checkpoint) if latest_checkpoint else 0
            backup_count     = extract_count(latest_backup) if latest_backup else 0

            if latest_backup and backup_count > checkpoint_count:
                print(f"  🚀 Using backup instead of checkpoint: {latest_backup.name}")
                with open(latest_backup, "r", encoding="utf-8") as f:
                    samples = json.load(f)

                return samples, len(samples), None, None, len(samples)

            elif latest_checkpoint:
                print(f"  📂 Loading checkpoint: {latest_checkpoint.name}")
                with open(latest_checkpoint, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)

                return (
                    checkpoint["samples"],
                    checkpoint["completed_count"],
                    checkpoint["type_schedule"],
                    checkpoint["instruction_list"],
                    checkpoint["total_target"]
                )

            return None, 0, None, None, 0

        except Exception as e:
            print(f"  ⚠️ Failed to load checkpoint: {e}")
            return None, 0, None, None, 0

    def _cleanup_old_checkpoints(self):
        checkpoints = sorted(Path(self.checkpoint_dir).glob("checkpoint_*.json"))
        for old in checkpoints[:-5]:
            old.unlink()

# ============================================================
# CREDIT TRACKER  (thread-safe)
# ============================================================

class CreditTracker:
    COST_PER_1M_INPUT_TOKENS  = 0.28
    COST_PER_1M_OUTPUT_TOKENS = 0.42

    def __init__(self, max_budget_usd: float = 10.0):
        self.max_budget_usd        = max_budget_usd
        self.total_input_tokens    = 0
        self.total_output_tokens   = 0
        self.total_cost            = 0.0
        self.credit_limit_reached  = False
        self._lock                 = threading.Lock()

    def track_usage(self, usage: Dict) -> bool:
        if not usage:
            return True
        with self._lock:
            input_tokens  = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            self.total_input_tokens  += input_tokens
            self.total_output_tokens += output_tokens
            input_cost  = (input_tokens  / 1_000_000) * self.COST_PER_1M_INPUT_TOKENS
            output_cost = (output_tokens / 1_000_000) * self.COST_PER_1M_OUTPUT_TOKENS
            self.total_cost += input_cost + output_cost

            if self.total_cost >= self.max_budget_usd * 0.95:
                print(f"\n  ⚠️  WARNING: Used ${self.total_cost:.2f}/${self.max_budget_usd:.2f} "
                      f"({(self.total_cost / self.max_budget_usd) * 100:.1f}%)")
            if self.total_cost >= self.max_budget_usd:
                self.credit_limit_reached = True
                print(f"  ❌ Credit limit reached! Stopping generation.")
                return False
        return True

    def get_remaining_budget(self) -> float:
        return max(0, self.max_budget_usd - self.total_cost)

    def estimate_samples_remaining(self, avg_cost_per_sample: float = 0.0003) -> int:
        return int(self.get_remaining_budget() / avg_cost_per_sample)

    def print_summary(self):
        print(f"\n  📊 Credit Usage Summary:")
        print(f"    Total cost:     ${self.total_cost:.4f}")
        print(f"    Budget:         ${self.max_budget_usd:.2f}")
        print(f"    Remaining:      ${self.get_remaining_budget():.4f}")
        print(f"    Input tokens:   {self.total_input_tokens:,}")
        print(f"    Output tokens:  {self.total_output_tokens:,}")
        print(f"    Total tokens:   {self.total_input_tokens + self.total_output_tokens:,}")

# ============================================================
# SAMPLE GENERATION  (prompt forces English output)
# ============================================================

def generate_single_sample(question_type: str, instruction_variant: str,
                            credit_tracker: Optional[CreditTracker] = None,
                            retry_count: int = 0) -> Optional[Dict]:
    context_data = select_context(question_type)
    context      = context_data["context"]
    question     = generate_question(context_data, question_type)
    instruction  = INSTRUCTION_VARIANTS[instruction_variant]

    if question_type == "answerable":
        type_instruction = (
            f'CRITICAL: This is an ANSWERABLE question. The answer MUST be in the context.\n'
            f'DO NOT output "{UNANSWERABLE}" for this question type.\n'
            f'Extract the EXACT answer from the context verbatim in English.\n'
            f'If you cannot find the answer, check the context again – it IS there.'
        )
    elif question_type == "unavailable":
        type_instruction = (
            f'CRITICAL: This question asks for information NOT in the context.\n'
            f'You MUST output EXACTLY: "{UNANSWERABLE}"\n'
            f'Do not try to infer or guess the answer.'
        )
    elif question_type == "multi_part":
        type_instruction = (
            'CRITICAL: This is a MULTI-PART question.\n'
            'Extract ALL relevant sentences from the context that answer each part in English.\n'
            'If some parts are not in context, answer only what is available.'
        )
    else:
        type_instruction = "Follow the rules below strictly. Answer in English."

    prompt = f"""Generate a QA sample with these specifications:

Instruction: {instruction}

Context: {context}

Question: {question}

Question Type: {question_type}

{type_instruction}

Rules:
- If answerable: Extract exact answer from context verbatim. NEVER say you cannot answer.
- If unavailable: Output exactly: "{UNANSWERABLE}"
- If multi_part: Extract ALL relevant sentences from context
- If short_learning: Extract only the definition sentence
- IMPORTANT: Answer entirely in English.

Return ONLY this JSON object (no markdown, no extra text, no explanation):
{{
    "instruction": "{instruction}",
    "input": "Context: {context}\\nQuestion: {question}",
    "output": "<answer here>",
    "metadata": {{
        "topic": "{context_data['topic']}",
        "question_type": "{question_type}",
        "difficulty": "medium",
        "context_type": "{context_data['type']}",
        "instruction_variant": "{instruction_variant}"
    }}
}}"""

    response, usage = call_api_with_retry(prompt, temperature=0.1)
    if not response:
        return None

    if credit_tracker and usage:
        credit_tracker.track_usage(usage)

    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        json_match = re.search(r'\{.*\}$', response, re.DOTALL)
        if json_match:
            response = json_match.group(0)

        sample = json.loads(response)

        if sample:
            if question_type == "answerable" and sample.get("output", "").strip() == UNANSWERABLE:
                if retry_count < 2:
                    print(f"    🔄 Answerable got unanswerable response, retrying...")
                    return generate_single_sample(question_type, instruction_variant,
                                                  credit_tracker, retry_count + 1)
                else:
                    print(f"    ⚠️ Answerable failed after {retry_count} retries")
                    return None

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
        print(f"  Response preview: {response[:300]}")
        return None
    except Exception as e:
        print(f"  Unexpected error: {e}")
        return None


def _worker(args) -> Tuple[int, Optional[Dict]]:
    idx, question_type, instruction_variant, credit_tracker = args
    try:
        sample = generate_single_sample(question_type, instruction_variant, credit_tracker)
        return idx, sample
    except Exception as e:
        print(f"  Worker error at idx {idx}: {e}")
        return idx, None

# ============================================================
# MAIN GENERATION LOOP  (concurrent)
# ============================================================

def generate_training_data(
    target_samples: int,
    resume: bool = False,
    max_budget_usd: float = 10.0,
    save_interval: int = 100,
    max_workers: int = 15,
) -> Tuple[List[Dict], CreditTracker]:

    print(f"\n{'='*60}")
    print(f"  Generating {target_samples} samples")
    print(f"  Workers: {max_workers} | Budget: ${max_budget_usd:.2f} | Resume: {resume}")
    print(f"{'='*60}\n")

    checkpoint_manager = CheckpointManager()
    credit_tracker     = CreditTracker(max_budget_usd=max_budget_usd)

    samples          = []
    start_idx        = 0
    type_schedule    = None
    instruction_list = None
    saved_target     = 0

    if resume:
        loaded_samples, loaded_idx, loaded_types, loaded_instructions, saved_target = \
            checkpoint_manager.load_latest_checkpoint()

        if loaded_samples is not None:
            samples          = loaded_samples
            start_idx        = loaded_idx
            type_schedule    = loaded_types
            instruction_list = loaded_instructions

            print(f"  ✓ Loaded {len(samples)} existing samples")
            print(f"  ✓ Continuing from schedule index {start_idx}")
        else:
            print("  ℹ️ No valid checkpoint found — starting fresh")
            resume = False

    if not resume or not samples:
        samples          = []
        start_idx        = 0
        type_schedule    = _type_schedule(target_samples)

        instruction_list = []
        for variant, weight in INSTRUCTION_WEIGHTS.items():
            instruction_list.extend([variant] * int(target_samples * weight))

        while len(instruction_list) < target_samples:
            instruction_list.append(random.choice(list(INSTRUCTION_VARIANTS.keys())))

        random.shuffle(instruction_list)

        checkpoint_manager.save_checkpoint(
            samples, 0, type_schedule, instruction_list, target_samples
        )

    if type_schedule is None or instruction_list is None:
        print("  ⚠️  Backup loaded without schedule metadata. Regenerating schedule...")

        type_schedule = _type_schedule(target_samples)

        instruction_list = []
        for variant, weight in INSTRUCTION_WEIGHTS.items():
            instruction_list.extend([variant] * int(target_samples * weight))

        while len(instruction_list) < target_samples:
            instruction_list.append(random.choice(list(INSTRUCTION_VARIANTS.keys())))

        random.shuffle(instruction_list)

    if saved_target != target_samples:
        print(f"  ⚠️  Target changed from {saved_target} to {target_samples}")

        if target_samples > len(type_schedule):
            extra = target_samples - len(type_schedule)

            type_schedule += _type_schedule(extra)
            instruction_list += [
                random.choice(list(INSTRUCTION_VARIANTS.keys()))
                for _ in range(extra)
            ]

    remaining_needed = target_samples - len(samples)
    if remaining_needed <= 0:
        print(f"  ✓ Already have {len(samples)} samples — nothing to do.")
        return samples, credit_tracker

    work_items = [
        (i, type_schedule[i], instruction_list[i], credit_tracker)
        for i in range(start_idx, target_samples)
    ]

    failed_count          = 0
    last_checkpoint_count = len(samples)
    completed_attempts    = 0
    total_work            = len(work_items)

    print(f"  Launching {total_work} work items across {max_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, item): item for item in work_items}

        for future in as_completed(futures):

            if credit_tracker.credit_limit_reached:
                print("\n  ❌ Budget limit — cancelling remaining futures...")
                for f in futures:
                    f.cancel()
                break

            idx, sample = future.result()
            completed_attempts += 1

            with _lock:
                if sample:
                    samples.append(sample)
                    status = f"✓  (total valid: {len(samples)})"
                else:
                    failed_count += 1
                    status = "✗"

                print(f"  [{completed_attempts}/{total_work}] idx={idx} {status}", flush=True)

                if completed_attempts % 50 == 0:
                    print(
                        f"\n  📊 {completed_attempts}/{total_work} attempts | "
                        f"{len(samples)} valid | "
                        f"${credit_tracker.total_cost:.4f} spent | "
                        f"~{credit_tracker.estimate_samples_remaining()} samples left in budget\n"
                    )

                if len(samples) - last_checkpoint_count >= save_interval:
                    checkpoint_manager.save_checkpoint(
                        samples, len(samples),
                        type_schedule, instruction_list,
                        target_samples
                    )
                    last_checkpoint_count = len(samples)

                if len(samples) >= target_samples:
                    print(f"\n  ✅ Reached target {target_samples} valid samples — stopping workers.")
                    for f in futures:
                        f.cancel()
                    break

    checkpoint_manager.save_checkpoint(
        samples, len(samples),
        type_schedule, instruction_list,
        target_samples
    )

    print(f"\n  ✓ Generation complete: {len(samples)} valid | {failed_count} failed")
    credit_tracker.print_summary()

    return samples, credit_tracker

# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_samples(samples: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Saved {len(samples)} samples → {path}")


def save_as_jsonl(samples: List[Dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            alpaca_format = {
                "instruction": sample.get("instruction", ""),
                "input":       sample.get("input", ""),
                "output":      sample.get("output", "")
            }
            f.write(json.dumps(alpaca_format, ensure_ascii=False) + "\n")
    print(f"  ✓ Saved {len(samples)} samples as JSONL → {path}")


def save_metadata(samples: List[Dict], credit_tracker: CreditTracker, path: str):
    metadata = {
        "total_samples":         len(samples),
        "total_cost_usd":        credit_tracker.total_cost,
        "total_input_tokens":    credit_tracker.total_input_tokens,
        "total_output_tokens":   credit_tracker.total_output_tokens,
        "budget_usd":            credit_tracker.max_budget_usd,
        "generation_date":       datetime.now().isoformat(),
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

# ============================================================
# QUALITY REPORT
# ============================================================

def quality_report(samples: List[Dict]) -> int:
    print(f"\n{'='*60}")
    print("  QUALITY REPORT")
    print(f"{'='*60}")

    type_counts        = Counter()
    instruction_counts = Counter()
    valid_count        = 0
    all_issues         = []

    for i, s in enumerate(samples):
        qtype = s.get("metadata", {}).get("question_type", "unknown")
        ivar  = s.get("metadata", {}).get("instruction_variant", "unknown")
        type_counts[qtype]        += 1
        instruction_counts[ivar]  += 1

        issues = []
        issues.extend(validate_sample(s))
        issues.extend(validate_multi_part(s))
        issues.extend(validate_type_consistency(s))

        if not issues:
            valid_count += 1
        else:
            all_issues.append({"sample_index": i, "qtype": qtype, "issues": issues})

    print(f"\n  Total samples:  {len(samples)}")
    print(f"  Valid samples:  {valid_count} ({valid_count / len(samples) * 100:.1f}%)")

    if all_issues:
        print(f"\n  ⚠️  Issues in {len(all_issues)} samples:")
        for issue in all_issues[:5]:
            print(f"    Sample {issue['sample_index'] + 1} ({issue['qtype']}):")
            for iss in issue["issues"]:
                print(f"      - {iss}")
        if len(all_issues) > 5:
            print(f"    ... and {len(all_issues) - 5} more")

    print(f"\n  Question Type Distribution:")
    for qtype, count in sorted(type_counts.items()):
        print(f"    {qtype:<15}: {count:>4} ({count / len(samples) * 100:.1f}%)")

    print(f"\n  Instruction Variant Distribution:")
    for ivar, count in sorted(instruction_counts.items()):
        print(f"    {ivar:<15}: {count:>4} ({count / len(samples) * 100:.1f}%)")

    return valid_count


def display_samples(samples: List[Dict], max_display: int = 5):
    print(f"\n{'='*60}")
    print(f"  SAMPLE PREVIEW (first {min(max_display, len(samples))})")
    print(f"{'='*60}")
    for i, s in enumerate(samples[:max_display]):
        print(f"\n  ── Sample {i + 1} ──")
        print(f"  Type:                {s.get('metadata', {}).get('question_type')}")
        print(f"  Instruction variant: {s.get('metadata', {}).get('instruction_variant')}")
        print(f"  Input:  {s.get('input', '')[:150]}...")
        print(f"  Output: {s.get('output', '')[:150]}...")


def test_single_generation():
    print("\n" + "=" * 60)
    print("  RUNNING SINGLE TEST GENERATION")
    print("=" * 60)
    sample = generate_single_sample("answerable", "formal")
    if sample:
        print("\n  ✓ Test successful!")
        print(json.dumps(sample, ensure_ascii=False, indent=2))
        return True
    print("\n  ✗ Test failed")
    return False

# ============================================================
# ENTRY POINT
# ============================================================

def main():
    global API_KEY
    env_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if env_key:
        API_KEY = env_key

    parser = argparse.ArgumentParser(
        description="Generate Nepali banking QA training data (concurrent + checkpoint)"
    )
    parser.add_argument("--samples",       "-n", type=int,   default=TEST_SAMPLES,
                        help="Number of valid samples to generate")
    parser.add_argument("--resume",        "-r", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--budget",        "-b", type=float, default=10.0,
                        help="Maximum budget in USD (default: 10.0)")
    parser.add_argument("--save-interval", "-s", type=int,   default=100,
                        help="Save checkpoint every N valid samples (default: 100)")
    parser.add_argument("--workers",       "-w", type=int,   default=15,
                        help="Concurrent API workers (default: 15). "
                             "Reduce to 8-10 if you see 429 rate-limit errors.")
    parser.add_argument("--test",          "-t", action="store_true",
                        help="Run a single test generation and exit")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  TRAINING DATA GENERATOR  (concurrent edition)")
    print(f"{'='*60}")

    if "sk-xxxxx" in API_KEY or API_KEY == "xxxx":
        print("\n  ❌ Please set your API key:")
        print("     export DEEPSEEK_API_KEY='sk-your-key-here'")
        print("  or edit API_KEY in the script.")
        return

    if args.test:
        test_single_generation()
        return

    samples, credit_tracker = generate_training_data(
        target_samples=args.samples,
        resume=args.resume,
        max_budget_usd=args.budget,
        save_interval=args.save_interval,
        max_workers=args.workers,
    )

    if not samples:
        print("\n  ❌ No samples generated")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path  = f"{OUTPUT_DIR}/training_data_{timestamp}_{len(samples)}_samples.json"
    jsonl_path = f"{OUTPUT_DIR}/training_data_{timestamp}_{len(samples)}_samples.jsonl"
    meta_path  = f"{OUTPUT_DIR}/metadata_{timestamp}_{len(samples)}_samples.json"

    save_samples(samples, json_path)
    save_as_jsonl(samples, jsonl_path)
    save_metadata(samples, credit_tracker, meta_path)
    display_samples(samples, min(5, len(samples)))
    quality_report(samples)

    if len(samples) < args.samples:
        print(f"\n{'='*60}")
        print("  ⚠️  PARTIAL GENERATION")
        print(f"{'='*60}")
        print(f"  Generated {len(samples)}/{args.samples} samples")
        print(f"  To resume:")
        script = os.path.basename(__file__)
        print(f"    python {script} --samples={args.samples} --resume "
              f"--workers={args.workers} --budget={args.budget}")
        print(f"  Checkpoints in: {CHECKPOINT_DIR}/")
    else:
        print(f"\n{'='*60}")
        print(f"  ✅ COMPLETE — {len(samples)} samples | ${credit_tracker.total_cost:.4f} spent")
        print(f"{'='*60}")

    print(f"\n  Files saved:")
    print(f"    {json_path}")
    print(f"    {jsonl_path}")
    print(f"    {meta_path}")


if __name__ == "__main__":
    main()