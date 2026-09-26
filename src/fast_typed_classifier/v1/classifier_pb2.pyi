from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class QuestionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    QUESTION_TYPE_UNSPECIFIED: _ClassVar[QuestionType]
    QUESTION_TYPE_CHOICE: _ClassVar[QuestionType]
    QUESTION_TYPE_SCORE: _ClassVar[QuestionType]
    QUESTION_TYPE_NOUL: _ClassVar[QuestionType]
QUESTION_TYPE_UNSPECIFIED: QuestionType
QUESTION_TYPE_CHOICE: QuestionType
QUESTION_TYPE_SCORE: QuestionType
QUESTION_TYPE_NOUL: QuestionType

class ClassifyRequest(_message.Message):
    __slots__ = ("request_id", "text", "json", "questions", "model", "lang", "max_len")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    JSON_FIELD_NUMBER: _ClassVar[int]
    QUESTIONS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    LANG_FIELD_NUMBER: _ClassVar[int]
    MAX_LEN_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    text: str
    json: str
    questions: _containers.RepeatedCompositeFieldContainer[Question]
    model: str
    lang: str
    max_len: int
    def __init__(self, request_id: _Optional[str] = ..., text: _Optional[str] = ..., json: _Optional[str] = ..., questions: _Optional[_Iterable[_Union[Question, _Mapping]]] = ..., model: _Optional[str] = ..., lang: _Optional[str] = ..., max_len: _Optional[int] = ...) -> None: ...

class Question(_message.Message):
    __slots__ = ("id", "instructions", "choice", "score", "noul")
    ID_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTIONS_FIELD_NUMBER: _ClassVar[int]
    CHOICE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    NOUL_FIELD_NUMBER: _ClassVar[int]
    id: str
    instructions: str
    choice: ChoiceQuestion
    score: ScoreQuestion
    noul: NoulQuestion
    def __init__(self, id: _Optional[str] = ..., instructions: _Optional[str] = ..., choice: _Optional[_Union[ChoiceQuestion, _Mapping]] = ..., score: _Optional[_Union[ScoreQuestion, _Mapping]] = ..., noul: _Optional[_Union[NoulQuestion, _Mapping]] = ...) -> None: ...

class ChoiceQuestion(_message.Message):
    __slots__ = ("options",)
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    options: _containers.RepeatedCompositeFieldContainer[ChoiceOption]
    def __init__(self, options: _Optional[_Iterable[_Union[ChoiceOption, _Mapping]]] = ...) -> None: ...

class ChoiceOption(_message.Message):
    __slots__ = ("label", "description")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    label: str
    description: str
    def __init__(self, label: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class ScoreQuestion(_message.Message):
    __slots__ = ("levels",)
    LEVELS_FIELD_NUMBER: _ClassVar[int]
    levels: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, levels: _Optional[_Iterable[str]] = ...) -> None: ...

class NoulQuestion(_message.Message):
    __slots__ = ("true_description", "false_description", "true_label", "false_label")
    TRUE_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    FALSE_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TRUE_LABEL_FIELD_NUMBER: _ClassVar[int]
    FALSE_LABEL_FIELD_NUMBER: _ClassVar[int]
    true_description: str
    false_description: str
    true_label: str
    false_label: str
    def __init__(self, true_description: _Optional[str] = ..., false_description: _Optional[str] = ..., true_label: _Optional[str] = ..., false_label: _Optional[str] = ...) -> None: ...

class ClassifyResponse(_message.Message):
    __slots__ = ("request_id", "answers", "routing", "input_tokens", "timing", "error")
    class AnswersEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Answer
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Answer, _Mapping]] = ...) -> None: ...
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    ANSWERS_FIELD_NUMBER: _ClassVar[int]
    ROUTING_FIELD_NUMBER: _ClassVar[int]
    INPUT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TIMING_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    answers: _containers.MessageMap[str, Answer]
    routing: Routing
    input_tokens: int
    timing: Timing
    error: Error
    def __init__(self, request_id: _Optional[str] = ..., answers: _Optional[_Mapping[str, Answer]] = ..., routing: _Optional[_Union[Routing, _Mapping]] = ..., input_tokens: _Optional[int] = ..., timing: _Optional[_Union[Timing, _Mapping]] = ..., error: _Optional[_Union[Error, _Mapping]] = ...) -> None: ...

class Answer(_message.Message):
    __slots__ = ("type", "choice", "score", "noul", "probabilities", "confidence", "answer_confidence", "act_probability")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    CHOICE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    NOUL_FIELD_NUMBER: _ClassVar[int]
    PROBABILITIES_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    ANSWER_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    ACT_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    type: QuestionType
    choice: str
    score: float
    noul: float
    probabilities: _containers.RepeatedScalarFieldContainer[float]
    confidence: float
    answer_confidence: float
    act_probability: float
    def __init__(self, type: _Optional[_Union[QuestionType, str]] = ..., choice: _Optional[str] = ..., score: _Optional[float] = ..., noul: _Optional[float] = ..., probabilities: _Optional[_Iterable[float]] = ..., confidence: _Optional[float] = ..., answer_confidence: _Optional[float] = ..., act_probability: _Optional[float] = ...) -> None: ...

class Routing(_message.Message):
    __slots__ = ("model", "reason", "detected_script", "detected_language")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    DETECTED_SCRIPT_FIELD_NUMBER: _ClassVar[int]
    DETECTED_LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    model: str
    reason: str
    detected_script: str
    detected_language: str
    def __init__(self, model: _Optional[str] = ..., reason: _Optional[str] = ..., detected_script: _Optional[str] = ..., detected_language: _Optional[str] = ...) -> None: ...

class Timing(_message.Message):
    __slots__ = ("queue_ms", "inference_ms", "batch_size")
    QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_MS_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    queue_ms: float
    inference_ms: float
    batch_size: int
    def __init__(self, queue_ms: _Optional[float] = ..., inference_ms: _Optional[float] = ..., batch_size: _Optional[int] = ...) -> None: ...

class Error(_message.Message):
    __slots__ = ("code", "message")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: int
    message: str
    def __init__(self, code: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class ClassifyBatchRequest(_message.Message):
    __slots__ = ("requests", "questions")
    REQUESTS_FIELD_NUMBER: _ClassVar[int]
    QUESTIONS_FIELD_NUMBER: _ClassVar[int]
    requests: _containers.RepeatedCompositeFieldContainer[ClassifyRequest]
    questions: _containers.RepeatedCompositeFieldContainer[Question]
    def __init__(self, requests: _Optional[_Iterable[_Union[ClassifyRequest, _Mapping]]] = ..., questions: _Optional[_Iterable[_Union[Question, _Mapping]]] = ...) -> None: ...

class ClassifyBatchResponse(_message.Message):
    __slots__ = ("responses",)
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    responses: _containers.RepeatedCompositeFieldContainer[ClassifyResponse]
    def __init__(self, responses: _Optional[_Iterable[_Union[ClassifyResponse, _Mapping]]] = ...) -> None: ...

class GetServerInfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ServerInfo(_message.Message):
    __slots__ = ("device", "models", "max_batch_rows", "max_batch_tokens", "max_inflight", "laya_version", "torch_version")
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_ROWS_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCH_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_INFLIGHT_FIELD_NUMBER: _ClassVar[int]
    LAYA_VERSION_FIELD_NUMBER: _ClassVar[int]
    TORCH_VERSION_FIELD_NUMBER: _ClassVar[int]
    device: str
    models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    max_batch_rows: int
    max_batch_tokens: int
    max_inflight: int
    laya_version: str
    torch_version: str
    def __init__(self, device: _Optional[str] = ..., models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ..., max_batch_rows: _Optional[int] = ..., max_batch_tokens: _Optional[int] = ..., max_inflight: _Optional[int] = ..., laya_version: _Optional[str] = ..., torch_version: _Optional[str] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("name", "repo", "max_len")
    NAME_FIELD_NUMBER: _ClassVar[int]
    REPO_FIELD_NUMBER: _ClassVar[int]
    MAX_LEN_FIELD_NUMBER: _ClassVar[int]
    name: str
    repo: str
    max_len: int
    def __init__(self, name: _Optional[str] = ..., repo: _Optional[str] = ..., max_len: _Optional[int] = ...) -> None: ...
