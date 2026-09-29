import os
from typing import Dict, Any, List
import json
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq


class AssignmentGenerator:
    def __init__(self, groq_api_key: str, model_name: str | None = None):
        selected_model = model_name or os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=selected_model)
        self.logger = logging.getLogger(__name__)

    def generate_assignments(self, assignment_request: Dict[str, Any]) -> Dict[str, Any]:
        student_id = assignment_request.get('student_id')
        topic = assignment_request.get('topic')
        difficulty_level = assignment_request.get('difficulty_level', 'Intermediate')
        student_stage = assignment_request.get('student_stage', 1)
        assignment_count = assignment_request.get('assignment_count', 3)
        focus_areas = assignment_request.get('focus_areas', [])
        assignment_types = assignment_request.get('assignment_types', ['mixed'])

        prompt = self._create_assignment_prompt(
            topic, difficulty_level, student_stage, assignment_count, focus_areas, assignment_types
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            parsed = self._parse_assignments(response.content, assignment_count)
        except Exception as exc:
            self.logger.warning("Assignment generation failed, using fallback data: %s", exc)
            parsed = self._fallback_assignments(topic, difficulty_level, assignment_count, focus_areas)

        final = {
            'student_id': student_id,
            'topic': topic,
            'difficulty_level': difficulty_level,
            'student_stage': student_stage,
            'generated_at': datetime.utcnow().isoformat(),
            'assignments': parsed,
            'assignment_count': len(parsed),
            'estimated_completion_time': f"{max(15, len(parsed) * 20)} min",
            'learning_objectives': self._build_learning_objectives(topic, difficulty_level, focus_areas),
            'focus_areas': focus_areas,
            'assignment_types': assignment_types,
        }
        return final

    def _create_assignment_prompt(self, topic: str, difficulty_level: str, student_stage: int,
                                  assignment_count: int, focus_areas: List[str], assignment_types: List[str]) -> str:
        return f"""
        Create a set of personalized learning assignments for a student.

        Topic: {topic}
        Difficulty: {difficulty_level}
        Student stage: {student_stage}
        Assignment count: {assignment_count}
        Focus areas: {', '.join(focus_areas) if focus_areas else 'General topic review'}
        Assignment types: {', '.join(assignment_types) if assignment_types else 'mixed'}

        Return valid JSON with this shape:
        {{
          "assignments": [
            {{
              "title": "Assignment title",
              "type": "reading|practice|project|analysis|reflection",
              "difficulty": "Beginner|Intermediate|Advanced",
              "estimated_time": "20 min",
              "description": "Short task description",
              "instructions": ["Step 1", "Step 2"],
              "deliverables": ["Result 1", "Result 2"],
              "rubric": ["Criterion 1", "Criterion 2"]
            }}
          ]
        }}
        Keep assignments practical, clear, and aligned to the topic.
        """

    def _parse_assignments(self, raw_response: str, assignment_count: int) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(raw_response)
            assignments = payload.get('assignments', []) if isinstance(payload, dict) else []
            if assignments:
                return assignments[:assignment_count]
        except Exception:
            pass

        try:
            start = raw_response.find('[')
            end = raw_response.rfind(']')
            if start != -1 and end != -1 and end > start:
                payload = json.loads(raw_response[start:end+1])
                if isinstance(payload, list):
                    return payload[:assignment_count]
        except Exception:
            pass

        return self._fallback_assignments('General Topic', 'Intermediate', assignment_count, [])

    def _fallback_assignments(self, topic: str, difficulty_level: str, assignment_count: int, focus_areas: List[str]) -> List[Dict[str, Any]]:
        focus = focus_areas or ["core concepts", "application"]
        tasks = [
            {
                'title': f'{topic} Concept Review',
                'type': 'reading',
                'difficulty': difficulty_level,
                'estimated_time': '20 min',
                'description': f'Review the key ideas and definitions for {topic} and summarize them in your own words.',
                'instructions': [
                    f'List the most important concepts in {topic}.',
                    'Write a short explanation for each concept.',
                    'Note any questions you still have.'
                ],
                'deliverables': ['Concept summary', 'Questions list'],
                'rubric': ['Accuracy', 'Clarity', 'Completion']
            },
            {
                'title': f'{topic} Applied Practice',
                'type': 'practice',
                'difficulty': difficulty_level,
                'estimated_time': '30 min',
                'description': f'Apply the main ideas of {topic} to a small practical task or example.',
                'instructions': [
                    'Choose one task related to the target topic.',
                    'Solve or explain the task using the concepts you learned.',
                    'Describe the method you used.'
                ],
                'deliverables': ['Completed activity', 'Short reflection'],
                'rubric': ['Correct application', 'Reasoning', 'Reflection']
            },
            {
                'title': f'{topic} Reflection and Improvement',
                'type': 'reflection',
                'difficulty': difficulty_level,
                'estimated_time': '15 min',
                'description': f'Reflect on your understanding and identify the next skill you need to improve in {topic}.',
                'instructions': [
                    'Describe what you understand well.',
                    'List one area that still feels difficult.',
                    'Set one small improvement goal for next time.'
                ],
                'deliverables': ['Reflection notes', 'Improvement plan'],
                'rubric': ['Honest assessment', 'Action plan', 'Depth of reflection']
            },
        ]
        if assignment_count <= 0:
            return []
        return tasks[:assignment_count]

    def _build_learning_objectives(self, topic: str, difficulty_level: str, focus_areas: List[str]) -> List[str]:
        objectives = [
            f'Explain the core concepts behind {topic}.',
            f'Apply {topic} in a simple real-world scenario.',
            f'Identify and improve weak areas related to {focus_areas[0] if focus_areas else "the topic"}.'
        ]
        if difficulty_level.lower() in {'intermediate', 'advanced'}:
            objectives.append(f'Analyze how {topic} connects to broader learning goals.')
        return objectives
