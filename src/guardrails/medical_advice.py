"""Medical advice detection — for healthcare safety"""
import re
from typing import Dict
from rich.console import Console

console = Console()


class MedicalAdviceDetector:
    """
    Detects when user is asking for personal medical advice.
    Adds stronger disclaimer in such cases.
    """
    
    # Keywords indicating personal medical advice request
    PERSONAL_PATTERNS = [
        r"\bshould i (take|use|stop|start)\b",
        r"\bdo i have\b",
        r"\bwhat should i do\b",
        r"\bdiagnose me\b",
        r"\bmy (symptoms?|diagnosis|condition|prescription)\b",
        r"\bi (am|have|feel)\b.*\b(pain|sick|symptom|disease)\b",
        r"\bcan you (prescribe|recommend|diagnose)\b",
        r"\bis it safe for me\b",
        r"\bwhat dosage should i\b",
    ]
    
    # Medical emergency keywords
    EMERGENCY_PATTERNS = [
        r"\b(emergency|urgent|critical|severe)\b.*\b(pain|bleeding|breathing|chest)\b",
        r"\bchest pain\b",
        r"\bheart attack\b",
        r"\bstroke\b",
        r"\bcan'?t breathe\b",
        r"\bunconscious\b",
        r"\bbleeding heavily\b",
    ]
    
    DISCLAIMER = (
        "\n\n---\n"
        "⚠️ **Medical Disclaimer:** Ye AI-generated information hai, "
        "sirf educational purpose ke liye. Ye professional medical advice, "
        "diagnosis, ya treatment ka substitute nahi hai. Apni health conditions "
        "ke liye qualified doctor se consult karein.\n"
    )
    
    EMERGENCY_DISCLAIMER = (
        "\n\n🚨 **EMERGENCY WARNING:** Agar aap medical emergency mein hain, "
        "turant 108 (India) ya apne local emergency number pe call karein. "
        "Ye AI assistant emergency medical care provide nahi kar sakta.\n"
    )
    
    def detect(self, query: str) -> Dict:
        """
        Detect if query is asking for personal medical advice.
        Returns detection info.
        """
        query_lower = query.lower()
        
        # Check emergency
        for pattern in self.EMERGENCY_PATTERNS:
            if re.search(pattern, query_lower):
                return {
                    "is_emergency": True,
                    "is_personal_advice": True,
                    "reason": "emergency_keywords",
                    "disclaimer": self.DISCLAIMER + self.EMERGENCY_DISCLAIMER,
                }
        
        # Check personal medical advice
        for pattern in self.PERSONAL_PATTERNS:
            if re.search(pattern, query_lower):
                return {
                    "is_emergency": False,
                    "is_personal_advice": True,
                    "reason": "personal_advice_pattern",
                    "disclaimer": self.DISCLAIMER,
                }
        
        # Default — educational query
        return {
            "is_emergency": False,
            "is_personal_advice": False,
            "reason": "educational_query",
            "disclaimer": self.DISCLAIMER,
        }


if __name__ == "__main__":
    console.print("\n[bold]🧪 Testing Medical Advice Detection[/bold]\n")
    
    detector = MedicalAdviceDetector()
    
    test_inputs = [
        "What are the side effects of metformin?",  # Educational
        "Should I take metformin for my diabetes?",  # Personal
        "I have chest pain, what should I do?",     # Emergency
        "Do I have diabetes based on my symptoms?",  # Personal
        "How does metformin work?",                  # Educational
    ]
    
    for text in test_inputs:
        result = detector.detect(text)
        status = "🚨 EMERGENCY" if result["is_emergency"] else (
            "⚠️ PERSONAL" if result["is_personal_advice"] else "✅ EDUCATIONAL"
        )
        console.print(f"[yellow]Query:[/yellow] {text}")
        console.print(f"[cyan]Status:[/cyan] {status}")
        console.print(f"[dim]Reason: {result['reason']}[/dim]\n")