"""
Smart Question Generator
Generate relevant questions based on document content.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm import get_chatgroq_model


def generate_insightful_questions(document_content, num_questions=3):
    """
    Generate insightful questions based on document content.
    
    Args:
        document_content (str): Document to analyze
        num_questions (int): Number of questions to generate (default: 3)
    
    Returns:
        list: List of generated questions
    """
    try:
        llm = get_chatgroq_model("Detailed")
        
        prompt = f"""Analyze this document and generate {num_questions} insightful, relevant questions that would help someone understand the key concepts better.

Requirements:
- Questions should be specific to the document content
- Questions should be thought-provoking and require understanding
- Questions should cover different aspects of the document
- Each question should be clear and concise
- Focus on important concepts, not trivial details

Format: Return only the questions, numbered 1., 2., 3.

DOCUMENT:
{document_content}

QUESTIONS:"""
        
        response = llm.invoke(prompt).content
        
        # Parse questions into a list
        questions = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering and clean up
                question = line.lstrip('0123456789.-•) ').strip()
                if question:
                    questions.append(question)
        
        return questions[:num_questions]
    
    except Exception as e:
        return [f"Error generating questions: {str(e)}"]


def generate_document_summary(document_content):
    """
    Generate a comprehensive summary of the document.
    
    Args:
        document_content (str): Document content
    
    Returns:
        str: Generated summary
    """
    try:
        llm = get_chatgroq_model("Detailed")
        
        prompt = f"""Create a comprehensive summary of this document.

Include:
- Main topic and purpose
- Key points and findings
- Important conclusions or recommendations
- Critical information

Keep the summary clear, concise, and well-structured.

DOCUMENT:
{document_content}

SUMMARY:"""
        
        response = llm.invoke(prompt).content
        return response
    
    except Exception as e:
        return f"Error generating summary: {str(e)}"