"""PII detection and redaction using Presidio"""
from typing import Dict, List, Tuple
from rich.console import Console

console = Console()


class PIIDetector:
    """
    Detect and redact PII using Microsoft Presidio.
    Handles: names, phone numbers, emails, credit cards, SSN, locations.
    """
    
    def __init__(self):
        console.print("[cyan]🔧 Loading Presidio...[/cyan]")
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        
        # Entities we care about (medical context)
        self.entities = [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "CREDIT_CARD",
            "US_SSN",
            "LOCATION",
            "DATE_TIME",
            "IP_ADDRESS",
        ]
        console.print("   ✅ Presidio loaded")
    
    def detect(self, text: str, language: str = "en") -> List[Dict]:
        """Detect PII entities in text"""
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities,
            language=language,
        )
        return [
            {
                "type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": text[r.start:r.end],
            }
            for r in results
        ]
    
    def redact(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Redact PII from text.
        Returns (redacted_text, detected_entities)
        """
        detections = self.detect(text)
        
        if not detections:
            return text, []
        
        # Anonymize
        from presidio_analyzer import RecognizerResult
        analyzer_results = [
            RecognizerResult(
                entity_type=d["type"],
                start=d["start"],
                end=d["end"],
                score=d["score"],
            )
            for d in detections
        ]
        
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,
        )
        
        return anonymized.text, detections


if __name__ == "__main__":
    console.print("\n[bold]🧪 Testing PII Detection[/bold]\n")
    
    detector = PIIDetector()
    
    test_inputs = [
        "Patient John Doe, aged 45, called from +1-555-123-4567 about diabetes.",
        "Contact me at jane.smith@hospital.com or 9876543210 for my report.",
        "My Aadhaar is 1234-5678-9012 and I live in Mumbai.",
        "What are the side effects of metformin?",  # No PII
    ]
    
    for text in test_inputs:
        console.print(f"[yellow]Input:[/yellow] {text}")
        redacted, detections = detector.redact(text)
        console.print(f"[green]Redacted:[/green] {redacted}")
        if detections:
            console.print(f"[cyan]Detected:[/cyan] {[d['type'] for d in detections]}")
        else:
            console.print("[dim]No PII detected[/dim]")
        console.print()