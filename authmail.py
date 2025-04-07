# Configuration and imports
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample email dataset
sample_emails = [
    {
        "id": "001",
        "from": "angry.customer@example.com",
        "subject": "Broken product received",
        "body": "I received my order #12345 yesterday but it arrived completely damaged. This is unacceptable and I demand a refund immediately. This is the worst customer service I've experienced.",
        "timestamp": "2024-03-15T10:30:00Z" 
    },
    {
        "id": "002",
        "from": "curious.shopper@example.com",
        "subject": "Question about product specifications",
        "body": "Hi, I'm interested in buying your premium package but I couldn't find information about whether it's compatible with Mac OS. Could you please clarify this? Thanks!",
        "timestamp": "2024-03-15T11:45:00Z"
    },
    {
        "id": "003",
        "from": "happy.user@example.com",
        "subject": "Amazing customer support",
        "body": "I just wanted to say thank you for the excellent support I received from Sarah on your team. She went above and beyond to help resolve my issue. Keep up the great work!",
        "timestamp": "2024-03-15T13:15:00Z"
    },
    {
        "id": "004",
        "from": "tech.user@example.com",
        "subject": "Need help with installation",
        "body": "I've been trying to install the software for the past hour but keep getting error code 5123. I've already tried restarting my computer and clearing the cache. Please help!",
        "timestamp": "2024-03-15T14:20:00Z"
    },
    {
        "id": "005",
        "from": "business.client@example.com",
        "subject": "Partnership opportunity",
        "body": "Our company is interested in exploring potential partnership opportunities with your organization. Would it be possible to schedule a call next week to discuss this further?",
        "timestamp": "2024-03-15T15:00:00Z"
    }
]


class EmailProcessor:
    def __init__(self):
        """Initialize the email processor with OpenAI API key."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define valid categories
        self.valid_categories = {
            "complaint", "inquiry", "feedback",
            "support_request", "other"
        }

    def classify_email(self, email: Dict) -> Optional[str]:
        """
        Classify an email using LLM.
        Returns the classification category or None if classification fails.
        
        TODO: 
        1. Design and implement the classification prompt
        2. Make the API call with appropriate error handling
        3. Validate and return the classification
        """

    def validate_email(self, email: Dict) -> bool:
        required_keys = {"id", "from", "subject", "body", "timestamp"}
        if not isinstance(email, dict):
            logger.error("Email is not a dictionary")
            return False
    
        missing = required_keys - email.keys()
        if missing:
            logger.error(f"Missing fields in email  {email.get('id', 'unknown')}: {missing} ")
            return False
        
        return True

    def classify_email(self, email:Dict) -> Optional [str]:

        
        if not self.validate_email(email):
            return None
        

        # Prompt instructing the LLM to classify into EXACTLY ONE of the valid categories

        prompt = (
                "You are a customer service AI expert that must classify incoming emails into EXACTLY ONE of the following categories\n"
                "- complaint: customer is expressing dissatisfaction or frustration\n "
                "- inquiry: asking for information or clarification\n "
                "- feedback:giving positive or neutral opinions\n "
                "- support_request: asking for help with using a product or service\n"
                "- other: doest not fit any of the above\n"
                "Return ONLY the category name\n"
                f"Subject: {email['subject']}\n"
                f"Body: {email['body']}"
        )

        # Extrac the category from the response 

        #Make the API call to the chat endpoint of GPT-3.5
        try: 
            response = self.client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages=[{"role":"user", "content": prompt}],
                temperature=0,
            )

            category = response.choices[0].message.content.strip().lower()


        # Check if the category is valid 

            if category in self.valid_categories:
                logger.info(f"Email {email['id']} classified as {category}")
                return category
            
            else:
                logger.warning(f"Invalid category returned by LLM: {category}")
                return "other"

        except Exception as e:
            logger.error(f"Error classifying email {email.get('id', 'unknown')}: {e}")
            return None


    def generate_response(self, email: Dict, classification: str) -> Optional[str]:
        """
        Generate an automated response based on email classification.
        
        TODO:
        1. Design the response generation prompt
        2. Implement appropriate response templates
        3. Add error handling
        """
        # Validate the email structure

        if not self.validate_email(email):
            return None
        
        # prompt instructing the LLM to generate a fully reply
        prompt = (
                f"You are a helpful professional customer support assistant."
                f"Based on the classification of the email and its content, write a full email response.\n\n"
                f"Classification: {classification}\n"
                f"Subject: {email['subject']}\n"
                f"Body: {email['body']}\n"
                "Respond with the full reply only. Be clear and empathetic"
        )
        
        try:  #call the openai chat completion api with the above prompt
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )

            # extract the AI gerenated reply
            reply = response.choices[0].message.content.strip()
            logger.info(f"Response generated for email {email['id']}")
            return reply


        except Exception as e:
            logger.info(f"Error generating response for email {email.get('id', 'unknown')}: {e}")
            return None


class EmailAutomationSystem:
    def __init__(self, processor: EmailProcessor):
        """Initialize the automation system with an EmailProcessor."""
        self.processor = processor
        self.response_handlers = {
            "complaint": self._handle_complaint,
            "inquiry": self._handle_inquiry,
            "feedback": self._handle_feedback,
            "support_request": self._handle_support_request,
            "other": self._handle_other
        }

    def process_email(self, email: Dict) -> Dict:
        """
        Process a single email through the complete pipeline.
        Returns a dictionary with the processing results.
        
        TODO:
        1. Implement the complete processing pipeline
        2. Add appropriate error handling
        3. Return processing results
        """
        try:
            
            # validate email

            if not self.processor.validate_email(email):
                return {
                    "email_id": email.get("id", "unknown"),
                    "success": False,
                    "classification": None,
                    "response_sent": False
                }

            # classify email

            classification = self.processor.classify_email(email)
            if not classification:
                return {
                    "email_id": email.get("id", "unknown"),
                    "success": False,
                    "classification": None,
                    "response_sent": False
                }

            # generate response

            response = self.processor.generate_response(email, classification)
            if not response:
                return {
                    "email_id": email["id"],
                    "success": False,
                    "classification": classification,
                    "response_sent": False
                }

            # call the specific category handler

            handler = self.response_handlers.get(classification, self._handle_other)
            handler(email, response)

            # return success record, including the generated response

            return {
                    "email_id": email["id"],
                    "success": True,
                    "classification": classification,
                    "response_sent": True,
                    "response_text": response
            }


        except Exception as e:
            logger.error(f"Error processing email {email.get('id', 'unknown')}: {e}")
            return {
                    "email_id": email.get('id', 'unknown'),
                    "success": False,
                    "classification": None,
                    "response_sent": False
            }

    def _handle_complaint(self, email: Dict, response: str):
        """
        Handle complaint emails.
        TODO: Implement complaint handling logic
        """
        send_complaint_response(email["id"], response)
        create_urgent_ticket(email["id"], "complaint", email["body"])

    def _handle_inquiry(self, email: Dict, response: str):
        """
        Handle inquiry emails.
        TODO: Implement inquiry handling logic
        """
        send_standard_response(email["id"], response)

    def _handle_feedback(self, email: Dict, response: str):
        """
        Handle feedback emails.
        TODO: Implement feedback handling logic
        """
        send_standard_response(email["id"], response)
        log_customer_feedback(email["id"], email["body"])

    def _handle_support_request(self, email: Dict, response: str):
        """
        Handle support request emails.
        TODO: Implement support request handling logic
        """
        send_standard_response(email["id"], response)
        create_support_ticket(email["id"], email["body"])

    def _handle_other(self, email: Dict, response: str):
        """
        Handle other category emails.
        TODO: Implement handling logic for other categories
        """
        send_standard_response(email["id"], response)

# Mock service functions
def send_complaint_response(email_id: str, response: str):
    """Mock function to simulate sending a response to a complaint"""
    logger.info(f"Sending complaint response for email {email_id}")
    # In real implementation: integrate with email service


def send_standard_response(email_id: str, response: str):
    """Mock function to simulate sending a standard response"""
    logger.info(f"Sending standard response for email {email_id}")
    # In real implementation: integrate with email service


def create_urgent_ticket(email_id: str, category: str, context: str):
    """Mock function to simulate creating an urgent ticket"""
    logger.info(f"Creating urgent ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def create_support_ticket(email_id: str, context: str):
    """Mock function to simulate creating a support ticket"""
    logger.info(f"Creating support ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def log_customer_feedback(email_id: str, feedback: str):
    """Mock function to simulate logging customer feedback"""
    logger.info(f"Logging feedback for email {email_id}")
    # In real implementation: integrate with feedback system


def run_demonstration():
    """Run a demonstration of the complete system."""
    # Initialize the system
    processor = EmailProcessor()
    automation_system = EmailAutomationSystem(processor)

    # Process all sample emails
    results = []
    for email in sample_emails:
        logger.info(f"\nProcessing email {email['id']}...")
        result = automation_system.process_email(email)
        results.append(result)

    # Create a summary DataFrame
    df = pd.DataFrame(results)
    print("\nProcessing Summary:")
    print(df[["email_id", "success", "classification", "response_sent", "response_text"]])

    return df


# Example usage:
if __name__ == "__main__":
    results_df = run_demonstration()