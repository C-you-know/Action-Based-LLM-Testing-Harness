# knit_space/tests/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union, Callable
from pathlib import Path
import uuid
import logging 
from contextlib import contextmanager
import json

@dataclass
class QAItem:
    """
    Represents a single QA item.
    """
    id: str
    question: Union[str, Dict[str, Any]] 
    answer: Any
    skill_coefficient: float
    modality: str 
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    verification_fn: Optional[Callable[[Any, Any], bool]] = field(default=None, compare=False)
    
    def verify(self, provided_answer: Any) -> bool:
        """Verify if provided answer matches expected answer"""
        if self.verification_fn:
            return self.verification_fn(self.answer, provided_answer,self)
        # Default exact match
        return self.answer == provided_answer
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (excluding verification_fn)"""
        return {
            'id': self.id,
            'question': self.question,
            'answer': self.answer,
            'modality': self.modality,
            'metadata': self.metadata
        }

class GenerationError(Exception):
    """Raised when test generation fails"""
    pass

class AbstractQATest(ABC):
    """
    Base class for QA test generators. Subclasses implement `generate()`
    and may accept parameters to slightly modify or augment questions.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # NEW: Generation statistics
        self._stats = {
            'total_generated': 0,
            'errors': 0,
            'last_generation_time': None
        }

    @property
    def name(self) -> str:
        """Identifier for the test generator"""
        return self.__class__.__name__

    @property
    def supported_modalities(self) -> List[str]:
        """Modalities this test supports by default"""
        return ['text']
    
    def read_file(self, path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """
        Utility to read a file into a string with better error handling.
        """
        try:
            file_path = Path(path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            
            with file_path.open('r', encoding=encoding) as f:
                content = f.read()
                self.logger.debug(f"Read {len(content)} characters from {path}")
                return content
        except Exception as e:
            self.logger.error(f"Error reading file {path}: {e}")
            raise GenerationError(f"Failed to read file {path}: {e}")
    
    def read_text_file(self, path: str) -> str:
        """Legacy method for backward compatibility"""
        return self.read_file(path)
    
    @contextmanager
    def resource_context(self):
        """Context manager for resource cleanup"""
        try:
            self.setup_resources()
            yield self
        finally:
            self.cleanup_resources()
    
    def setup_resources(self):
        """Override to setup resources (DB connections, models, etc.)"""
        pass
    
    def cleanup_resources(self):
        """Override to cleanup resources"""
        pass
    
    def validate_item(self, item: QAItem) -> bool:
        """
        Validate a generated QA item. Override for custom validation.
        """
        if not item.id or not item.question:
            return False
        if item.modality not in self.supported_modalities:
            return False
        return True
    
    def build_question(self, 
                      base_question: str,
                      prefix: Optional[str] = None,
                      suffix: Optional[str] = None, 
                      text_file: Optional[str] = None,
                      template_vars: Optional[Dict[str, str]] = None) -> str:
        """
        Build a complete question from components with template support.
        """
        # Apply template variables if provided
        if template_vars:
            try:
                base_question = base_question.format(**template_vars)
            except KeyError as e:
                self.logger.warning(f"Template variable missing: {e}")
        
        # Load file content if provided
        file_text = self.read_file(text_file) if text_file else ''
        
        # Build full question
        pieces = [prefix or '', base_question, suffix or '', file_text]
        return ' '.join([p for p in pieces if p]).strip()
    
    @abstractmethod
    def generate(self,
                 count: int = 1,
                 difficulty: Optional[str] = None,
                 prefix: Optional[str] = None,
                 suffix: Optional[str] = None,
                 text_file: Optional[str] = None,
                 template_vars: Optional[Dict[str, str]] = None,
                 **kwargs) -> Iterator[QAItem]:
        """
        Yield QAItem instances.

        Args:
            count: number of items to generate
            difficulty: optional difficulty tag
            prefix: text to prepend to each question
            suffix: text to append to each question
            text_file: path to file whose contents are appended to the question
            template_vars: dictionary for template variable substitution
            **kwargs: other custom parameters for specific generators
        """
        pass
    
    def generate_safe(self, **kwargs) -> List[QAItem]:
        """
        Generate items with comprehensive error handling and validation.
        """
        items = []
        errors = []
        
        try:
            with self.resource_context():
                for item in self.generate(**kwargs):
                    try:
                        if self.validate_item(item):
                            items.append(item)
                            self._stats['total_generated'] += 1
                        else:
                            errors.append(f"Invalid item: {item.id}")
                            self._stats['errors'] += 1
                    except Exception as e:
                        errors.append(f"Error processing item: {e}")
                        self._stats['errors'] += 1
                        
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise GenerationError(f"Generation failed for {self.name}: {e}")
        
        if errors:
            self.logger.warning(f"Generated {len(items)} items with {len(errors)} errors")
            
        return items
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self._stats.copy()

class TestRegistry:
    """Registry for QA test classes with filtering capabilities"""
    
    def __init__(self):
        self._tests: Dict[str, type] = {}
        self._tags: Dict[str, List[str]] = {}  # test_name -> [tags]
    
    def register(self, test_class: type, tags: Optional[List[str]] = None):
        """Register a test class with optional tags"""
        name = test_class.__name__
        self._tests[name] = test_class
        self._tags[name] = tags or []
    
    def get_by_tag(self, tag: str) -> List[type]:
        """Get all test classes with a specific tag"""
        return [self._tests[name] for name, tags in self._tags.items() if tag in tags]
    
    def get_by_modality(self, modality: str) -> List[type]:
        """Get all test classes supporting a modality"""
        result = []
        for test_class in self._tests.values():
            # Need to instantiate to check supported_modalities
            instance = test_class()
            if modality in instance.supported_modalities:
                result.append(test_class)
        return result
    
    def list_all(self) -> List[type]:
        """Get all registered test classes"""
        return list(self._tests.values())

test_registry = TestRegistry()

def register_test(*tags):
    """Decorator to register test classes"""
    def decorator(cls):
        test_registry.register(cls, list(tags))
        return cls
    return decorator

def create_test_cases(test_classes_or_instances: List[Union[type, object]],
                      config: Optional[Dict[str, Any]] = None,
                      **gen_kwargs) -> tuple[List[QAItem], List[Any]]:
    """
    EXACT API PRESERVED: Instantiate each class and collect QAItems and answers.
    Now supports both classes and pre-configured instances.
    Passes prefix/suffix/text_file etc via gen_kwargs.
    Returns (list of QAItem, list of answers).
    """
    items: List[QAItem] = []
    stats = {'total_processed': len(test_classes_or_instances), 'successful': 0, 'failed': 0}
    
    for test_item in test_classes_or_instances:
        try:
            # Handle both classes and instances
            if isinstance(test_item, type):
                # It's a class, instantiate it
                instance = test_item(config)
            else:
                # It's already an instance
                instance = test_item
            
            # Use safe generation with validation
            generated_items = instance.generate_safe(**gen_kwargs)
            items.extend(generated_items)
            stats['successful'] += 1
            
        except Exception as e:
            logging.error(f"Failed to generate from {test_item}: {e}")
            stats['failed'] += 1
    
    logging.info(f"Generated {len(items)} items from {stats['successful']}/{stats['total_processed']} tests")
    
    # EXACT RETURN FORMAT PRESERVED
    questions = items  # QAItem objects (backward compatible)
    answers = [item.answer for item in items]
    return questions, answers

def create_test_cases_advanced(test_classes_or_instances: Optional[List[Union[type, object]]] = None,
                              config: Optional[Dict[str, Any]] = None,
                              batch_size: Optional[int] = None,
                              filter_fn: Optional[Callable[[QAItem], bool]] = None,
                              save_to: Optional[str] = None,
                              use_registry: bool = False,
                              tags: Optional[List[str]] = None,
                              **gen_kwargs) -> tuple[List[QAItem], List[Any]]:
    """
    Advanced test case creation with additional features.
    """
    # Determine test classes to use
    if use_registry:
        if tags:
            test_classes = []
            for tag in tags:
                test_classes.extend(test_registry.get_by_tag(tag))
        else:
            test_classes = test_registry.list_all()
    else:
        test_classes = test_classes_or_instances or []
    
    all_items: List[QAItem] = []
    stats = {'total_classes': len(test_classes), 'successful': 0, 'failed': 0}
    
    for test_item in test_classes:
        try:
            # Handle both classes and instances
            if isinstance(test_item, type):
                instance = test_item(config)
            else:
                instance = test_item
            
            if batch_size:
                # Process in batches
                remaining = gen_kwargs.get('count', 1)
                while remaining > 0:
                    current_batch = min(batch_size, remaining)
                    batch_kwargs = {**gen_kwargs, 'count': current_batch}
                    
                    items = instance.generate_safe(**batch_kwargs)
                    if filter_fn:
                        items = [item for item in items if filter_fn(item)]
                    
                    all_items.extend(items)
                    remaining -= current_batch
            else:
                items = instance.generate_safe(**gen_kwargs)
                if filter_fn:
                    items = [item for item in items if filter_fn(item)]
                all_items.extend(items)
            
            stats['successful'] += 1
            
        except Exception as e:
            logging.error(f"Failed to generate from {test_item}: {e}")
            stats['failed'] += 1
    
    # Save results if requested
    if save_to:
        try:
            with open(save_to, 'w') as f:
                json.dump([item.to_dict() for item in all_items], f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save results: {e}")
    
    questions = all_items
    answers = [item.answer for item in all_items]
    
    logging.info(f"Generated {len(all_items)} items from {stats['successful']}/{stats['total_classes']} classes")
    
    return questions, answers

