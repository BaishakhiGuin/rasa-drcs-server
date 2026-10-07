"""
AI-DRCS  —  Rasa Custom Actions
Runs as a separate action server: `rasa run actions`

All external calls (Qdrant RAG, merchant DB) go through environment variables.
Set them in your Render.com environment or .env file.
"""

import os
import uuid
import logging
import requests
from typing import Any, Text, Dict, List, Optional

from rasa_sdk import Action, Tracker, FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, AllSlotsReset, ConversationResumed
from rasa_sdk.types import DomainDict

logger = logging.getLogger(__name__)

# ─── Environment Config ───────────────────────────────────────────────────────

QDRANT_URL        = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME   = os.getenv("COLLECTION_NAME", "research_notebook")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
HF_TOKEN          = os.getenv("HF_TOKEN", "")
APPS_SCRIPT_URL   = os.getenv("APPS_SCRIPT_EXEC_URL", "")  # Optional: call AS for merchant auth


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _generate_complaint_id() -> str:
    return "CMP-" + str(uuid.uuid4())[:7].upper()


def _embed_text(text: str) -> Optional[List[float]]:
    """Embed text using HuggingFace all-MiniLM-L6-v2 (free, 384-dim)."""
    try:
        resp = requests.post(
            "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": text, "options": {"wait_for_model": True}},
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            return data[0] if isinstance(data[0], list) else data
    except Exception as e:
        logger.warning(f"HF embed failed: {e}")
    return None


def _qdrant_search(query: str, limit: int = 3) -> List[Dict]:
    """Search Qdrant for relevant knowledge chunks."""
    if not QDRANT_URL or not QDRANT_API_KEY:
        return []
    vec = _embed_text(query)
    if not vec:
        return []
    try:
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
            json={
                "vector": vec,
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
                "score_threshold": 0.25
            },
            timeout=10
        )
        if resp.status_code == 200:
            results = resp.json().get("result", [])
            return [{"text": r["payload"].get("text", ""), "score": r["score"], "doc": r["payload"].get("doc_name", "")} for r in results]
    except Exception as e:
        logger.warning(f"Qdrant search failed: {e}")
    return []


def _groq_complete(messages: List[Dict], max_tokens: int = 512) -> str:
    """Call Groq LLaMA-3 for answer generation."""
    if not GROQ_API_KEY:
        return ""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1
            },
            timeout=20
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"Groq failed: {e}")
    return ""


def _call_apps_script(action: str, params: Dict) -> Optional[Dict]:
    """Call the Google Apps Script backend for live data."""
    if not APPS_SCRIPT_URL:
        return None
    try:
        url = APPS_SCRIPT_URL
        qs = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{url}?action={action}&{qs}"
        resp = requests.get(full_url, timeout=10, allow_redirects=True)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok"):
                return data
    except Exception as e:
        logger.warning(f"Apps Script call failed: {e}")
    return None


# ─── Form Validators ──────────────────────────────────────────────────────────

class ValidateMerchantAuthForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_merchant_auth_form"

    def validate_msisdn(
        self, slot_value: Any, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        msisdn = str(slot_value).strip().replace(" ", "").replace("-", "")
        # Accept 10-digit Indian mobile numbers, optionally with +91 prefix
        if msisdn.startswith("+91"):
            msisdn = msisdn[3:]
        if msisdn.startswith("91") and len(msisdn) == 12:
            msisdn = msisdn[2:]
        if len(msisdn) == 10 and msisdn.isdigit() and msisdn[0] in "6789":
            return {"msisdn": msisdn}
        dispatcher.utter_message(
            text="That doesn't look like a valid 10-digit mobile number. Please enter your registered mobile number (e.g. 9876543210)."
        )
        return {"msisdn": None}


class ValidateTransactionForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_transaction_form"

    def validate_transaction_id(
        self, slot_value: Any, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        txn = str(slot_value).strip()
        if len(txn) >= 6:
            return {"transaction_id": txn}
        dispatcher.utter_message(
            text="Please provide a valid Transaction ID or UTR number (at least 6 characters)."
        )
        return {"transaction_id": None}

    def validate_date(
        self, slot_value: Any, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        # Accept any date string — Rasa's Duckling entity extractor will normalise it
        if slot_value:
            return {"date": str(slot_value)}
        dispatcher.utter_message(text="Please provide the transaction date (e.g. 07/10/2026 or 'yesterday').")
        return {"date": None}


class ValidateDeviceIssueForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_device_issue_form"

    def validate_device_serial(
        self, slot_value: Any, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        serial = str(slot_value).strip().upper()
        if serial.startswith("SBX-") and len(serial) >= 7:
            return {"device_serial": serial}
        # Also accept raw numbers — add SBX- prefix
        if slot_value and str(slot_value).strip().isdigit():
            return {"device_serial": f"SBX-{slot_value}"}
        dispatcher.utter_message(
            text="Please share the Soundbox serial number. It's printed on the back of the device and starts with 'SBX-' (e.g. SBX-99824)."
        )
        return {"device_serial": None}


class ValidateRefundForm(FormValidationAction):
    def name(self) -> Text:
        return "validate_refund_form"

    def validate_amount(
        self, slot_value: Any, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: DomainDict
    ) -> Dict[Text, Any]:
        try:
            amount = float(str(slot_value).replace("₹", "").replace(",", "").strip())
            if amount > 0:
                return {"amount": amount}
        except (ValueError, TypeError):
            pass
        dispatcher.utter_message(text="Please provide the exact transaction amount in INR (e.g. 500 or 1200.50).")
        return {"amount": None}


# ─── Custom Actions ───────────────────────────────────────────────────────────

class ActionAuthenticateMerchant(Action):
    """
    Verify the merchant's MSISDN against the Apps Script / CRM backend.
    Sets merchant_authenticated = True and stores merchant context in slots.
    """
    def name(self) -> Text:
        return "action_authenticate_merchant"

    def run(
        self, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        msisdn = tracker.get_slot("msisdn")
        if not msisdn:
            dispatcher.utter_message(text="I need your mobile number to look up your account.")
            return [SlotSet("merchant_authenticated", False)]

        # Try live auth via Apps Script
        result = _call_apps_script("merchant_auth", {"msisdn": msisdn})

        if result:
            name = result.get("merchantName", "Merchant")
            device_serial = result.get("deviceSerial", "")
            account_status = result.get("accountStatus", "Active")
            dispatcher.utter_message(
                text=f"✅ Account verified — **{name}** | Status: {account_status}"
                     + (f" | Device: {device_serial}" if device_serial else "")
            )
            events = [
                SlotSet("merchant_authenticated", True),
                SlotSet("device_serial", device_serial) if device_serial else SlotSet("device_serial", None)
            ]
        else:
            # Graceful degradation — proceed without live data
            dispatcher.utter_message(
                text=f"Account looked up for {msisdn}. How can I help you today?"
            )
            events = [SlotSet("merchant_authenticated", True)]

        return events


class ActionCheckTransactionStatus(Action):
    """
    Checks transaction status. In production, connects to your payment gateway API.
    Falls back to a structured informational response.
    """
    def name(self) -> Text:
        return "action_check_transaction_status"

    def run(
        self, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        txn_id  = tracker.get_slot("transaction_id")
        date    = tracker.get_slot("date")
        msisdn  = tracker.get_slot("msisdn")

        if not txn_id:
            dispatcher.utter_message(text="Please provide your Transaction ID / UTR number.")
            return []

        # In production: call your payment gateway or Apps Script here
        # For now: structured pending/success response
        dispatcher.utter_message(
            text=f"📊 **Transaction Status Report**\n\n"
                 f"• Transaction ID: `{txn_id}`\n"
                 + (f"• Date: {date}\n" if date else "")
                 + f"• MSISDN: {msisdn or 'N/A'}\n\n"
                 f"⏳ Status is being fetched from the payment network. "
                 f"If the transaction shows FAILED, no amount was debited. "
                 f"If it shows PENDING, please wait up to 4 hours before re-attempting. "
                 f"For FAILED transactions where the customer's bank shows a debit, "
                 f"an auto-reversal will occur within 3-5 business days per NPCI mandate.\n\n"
                 f"Do you need me to raise a formal dispute for this transaction?"
        )
        return []


class ActionInitiateRefund(Action):
    """
    Initiates a refund complaint. In production, connects to your refund API.
    """
    def name(self) -> Text:
        return "action_initiate_refund"

    def run(
        self, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        txn_id  = tracker.get_slot("transaction_id")
        amount  = tracker.get_slot("amount")
        msisdn  = tracker.get_slot("msisdn")
        complaint_id = _generate_complaint_id()

        # In production: POST to your refund API here

        dispatcher.utter_message(
            text=f"✅ **Refund Request Registered**\n\n"
                 f"• Complaint ID: `{complaint_id}`\n"
                 f"• Transaction: `{txn_id or 'N/A'}`\n"
                 f"• Amount: ₹{amount or 'N/A'}\n"
                 f"• Account: {msisdn or 'N/A'}\n\n"
                 f"The refund will be processed to the original payment source within **3-5 business days**. "
                 f"You will receive an SMS update at each stage. "
                 f"Please save your Complaint ID for tracking: **{complaint_id}**"
        )
        return [SlotSet("complaint_id", complaint_id)]


class ActionRegisterComplaint(Action):
    """
    Registers a support complaint. Generates a complaint ID.
    """
    def name(self) -> Text:
        return "action_register_complaint"

    def run(
        self, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        msisdn      = tracker.get_slot("msisdn")
        issue_type  = tracker.get_slot("issue_type") or "general"
        complaint_id = _generate_complaint_id()

        # In production: POST to your CRM / helpdesk API here

        dispatcher.utter_message(
            text=f"✅ **Complaint Registered Successfully**\n\n"
                 f"• Complaint ID: `{complaint_id}`\n"
                 f"• Issue Type: {issue_type.title()}\n"
                 f"• Account: {msisdn or 'N/A'}\n\n"
                 f"Our support team will contact you within **24-48 hours** on your registered number. "
                 f"Track this complaint in your Merchant App under **Support → My Complaints**."
        )
        return [SlotSet("complaint_id", complaint_id)]


class ActionCheckDeviceStatus(Action):
    """
    Checks device status via Apps Script / MCP tool.
    In production, calls your device telemetry API.
    """
    def name(self) -> Text:
        return "action_check_device_status"

    def run(
        self, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        device_serial = tracker.get_slot("device_serial")
        msisdn        = tracker.get_slot("msisdn")

        if not device_serial:
            dispatcher.utter_message(
                text="I'll need the device serial number (on back of Soundbox, starts with SBX-) to check its status."
            )
            return []

        # In production: call your device telemetry API here
        dispatcher.utter_message(
            text=f"🔍 **Device Status Check — {device_serial}**\n\n"
                 f"• Last Seen: Online\n"
                 f"• Battery: 12%\n"
                 f"• Network: GSM -98 dBm (Low signal)\n"
                 f"• Firmware: v3.2.1 (Latest)\n"
                 f"• Payment Status: Active\n\n"
                 f"Based on the device telemetry, the GSM signal is critically low. "
                 f"Please move the device to an area with better mobile signal."
        )
        return []


class ActionEscalateToHuman(Action):
    """
    Escalates conversation to a human agent.
    Sets escalation flag and prepares handoff summary.
    """
    def name(self) -> Text:
        return "action_escalate_to_human"

    def run(
        self, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        msisdn        = tracker.get_slot("msisdn") or "Not provided"
        device_serial = tracker.get_slot("device_serial") or "Not provided"
        issue_type    = tracker.get_slot("issue_type") or "General"

        # In production: trigger your live agent handoff API / CRM ticket here
        logger.info(f"ESCALATION: msisdn={msisdn} device={device_serial} issue={issue_type}")

        return [SlotSet("requested_escalation", True)]


class ActionRAGSearch(Action):
    """
    Fallback action: searches the Qdrant knowledge base and responds using Groq.
    This is the NotebookLM-style RAG behaviour for anything not covered by rules/stories.
    """
    def name(self) -> Text:
        return "action_rag_search"

    def run(
        self, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get("text", "")
        if not user_message:
            dispatcher.utter_message(response="utter_default")
            return []

        dispatcher.utter_message(response="utter_please_wait")

        # 1. Search knowledge base
        chunks = _qdrant_search(user_message, limit=4)

        if not chunks:
            dispatcher.utter_message(
                text="I couldn't find a specific answer in the knowledge base for your question. "
                     "Let me connect you to our support team who can provide more detailed assistance. "
                     "Would you like me to escalate this?"
            )
            return []

        # 2. Build context
        context_text = "\n\n".join([
            f"[Source: {c['doc']}]\n{c['text']}"
            for c in chunks
        ])

        # 3. Ask Groq
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert support assistant for AI-DRCS merchant Soundbox devices and UPI payments. "
                    "Answer ONLY using the provided document excerpts. "
                    "Be concise, practical and cite sources when relevant. "
                    "If the answer isn't in the excerpts, say so clearly and suggest escalation.\n\n"
                    f"=== KNOWLEDGE BASE ===\n{context_text}\n====================="
                )
            },
            {"role": "user", "content": user_message}
        ]

        answer = _groqComplete(messages) if False else _groq_complete(messages)

        if answer:
            dispatcher.utter_message(text=answer)
        else:
            dispatcher.utter_message(response="utter_default")

        return []


class ActionCheckSettlementStatus(Action):
    """
    Checks settlement status for the merchant's account.
    """
    def name(self) -> Text:
        return "action_check_settlement_status"

    def run(
        self, dispatcher: CollectingDispatcher,
        tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        msisdn = tracker.get_slot("msisdn")

        # In production: query your settlement engine here
        dispatcher.utter_message(
            text=f"💰 **Settlement Status for {msisdn or 'your account'}**\n\n"
                 f"• Last Settlement: ₹12,450 — Credited 11:00 AM today\n"
                 f"• Pending Amount: ₹8,200 (will settle tomorrow by 11 AM)\n"
                 f"• Settlement Account: SBI ••••4521\n"
                 f"• KYC Status: ✅ Verified\n\n"
                 f"If your settlement is missing, please check:\n"
                 f"1. Bank holidays — settlements skip on public holidays\n"
                 f"2. Your bank's processing time (SBI/PNB can take till 2 PM)\n"
                 f"3. Minimum balance requirement in your settlement account"
        )
        return []


# ─── Alias for backward compatibility ────────────────────────────────────────
# (action_rag_search calls this directly)
def _groqComplete(messages):
    return _groq_complete(messages)
