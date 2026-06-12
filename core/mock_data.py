from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

from core.schemas import COGNITIVE_DOMAINS, DISCLAIMER


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "demo" / "fixtures"

FIXTURE_FILES = {
    "normal": "normal_user_sessions.json",
    "mild_decline": "mild_decline_sessions.json",
    "fluctuating": "fluctuating_user_sessions.json",
}

CLOCK_SAMPLE_DIR = PROJECT_ROOT / "assets" / "classroom_clock_samples"
CLOCK_SAMPLE_FILES = {
    "normal": "classroom_clock_normal.png",
    "spatial_shift": "classroom_clock_mild_decline.png",
    "wrong_hands": "classroom_clock_obvious_issue.png",
}

DIALOG_EXAMPLE_ANSWER_TYPES = ("normal", "mild_decline", "vague")
INTERVIEW_COMPLETED_MESSAGE = "主要认知域已覆盖，可以生成认知评估。"
CLASSROOM_DEMO_LEVEL_LABELS = ("正常表现", "轻度下降", "明显异常")
CLASSROOM_DEMO_RISK_TENDENCY = {
    "正常表现": "low",
    "轻度下降": "medium",
    "明显异常": "high",
}
CLASSROOM_CLOCK_REPORT_PROFILES = {
    "正常表现": {
        "risk_level": "low",
        "filename": "classroom_clock_normal.png",
        "domain_scores": {
            "executive_function": 0.92,
            "visuospatial": 0.94,
        },
        "clock_findings": {
            "number_placement": "数字完整、顺序基本正确，整体分布较均衡。",
            "hand_accuracy": "指针能表达 11:10，长短针方向清楚。",
            "visuospatial_evidence": [
                "数字围绕钟面分布较均衡。",
                "中心锚点清楚，指针指向目标时间。",
            ],
        },
        "cdt_features": {
            "numbers_complete": True,
            "number_order_correct": True,
            "number_spacing": "normal",
            "number_distribution": "balanced",
            "hands_present": True,
            "target_time_match": True,
            "center_anchor_clear": True,
        },
        "explanation": "演示示意画钟整体结构完整，未呈现明显视觉空间或执行步骤异常信号。",
    },
    "轻度下降": {
        "risk_level": "medium",
        "filename": "classroom_clock_mild_decline.png",
        "domain_scores": {
            "executive_function": 0.58,
            "visuospatial": 0.50,
        },
        "clock_findings": {
            "number_placement": "数字有轻度偏移，部分数字间距不均，右侧略集中。",
            "hand_accuracy": "指针方向接近目标时间，但长短针区分和目标时间匹配不够稳定。",
            "visuospatial_evidence": [
                "数字集中在右侧，空间布局略不均衡。",
                "指针能表达大致意图，但目标时间匹配不够准确。",
            ],
        },
        "cdt_features": {
            "numbers_complete": True,
            "number_order_correct": True,
            "number_spacing": "irregular",
            "number_distribution": "right_shifted",
            "hands_present": True,
            "target_time_match": False,
            "center_anchor_clear": True,
        },
        "explanation": "演示示意画钟出现轻度空间布局和指针匹配不稳定，和对话中的路线不确定共同构成需要关注的信号。",
    },
    "明显异常": {
        "risk_level": "high",
        "filename": "classroom_clock_obvious_issue.png",
        "domain_scores": {
            "executive_function": 0.34,
            "visuospatial": 0.28,
        },
        "clock_findings": {
            "number_placement": "数字明显聚集并有遗漏或顺序混乱，钟面空间布局不稳定。",
            "hand_accuracy": "指针目标时间不准确，中心锚点和长短针关系不清。",
            "visuospatial_evidence": [
                "数字集中在局部区域，整体空间组织困难。",
                "指针方向与 11:10 不匹配，执行目标时间要求困难。",
            ],
        },
        "cdt_features": {
            "numbers_complete": False,
            "number_order_correct": False,
            "number_spacing": "crowded",
            "number_distribution": "clustered",
            "hands_present": True,
            "target_time_match": False,
            "center_anchor_clear": False,
        },
        "explanation": "演示示意画钟呈现较明显的数字布局、空间组织和目标时间执行困难，和对话异常线索一起提示较高关注风险。",
    },
}
CLASSROOM_DEMO_CLOCK_TRIGGER_PLAN = {
    "轻度下降": {
        "after_turn": 6,
        "title": "建议进入画钟拍照环节",
        "elder_message": "我们再做一个小小游戏，好吗？请您在纸上画一个钟，指到 11 点 10 分。画好后拍张照片就可以，不着急，慢慢来。",
        "staff_message": "对话完成后，系统建议补充画钟拍照，用来观察空间布局和执行步骤。",
    },
    "明显异常": {
        "after_turn": 6,
        "title": "进入画钟拍照环节",
        "elder_message": "我们再做一个小小游戏，好吗？请您在纸上画一个钟，指到 11 点 10 分。画好后拍张照片就可以，不着急，慢慢来。",
        "staff_message": "多轮回答中出现连续不确定，访谈结束后再进入画钟拍照环节。",
    },
}
CLASSROOM_DEMO_DIALOGUES = {
    "正常表现": [
        {
            "domain": "orientation",
            "question": "{display_name}，您好，我是小顾，今天陪您轻松聊几句。您知道今天大概是星期几吗？",
            "answer": "今天是周六，我早上看了手机上的日期。",
        },
        {
            "domain": "memory",
            "question": "我先说三个词：梨子、雨伞、公交卡。您跟着说一遍可以吗？",
            "answer": "梨子、雨伞、公交卡，我记住了。",
        },
        {
            "domain": "language",
            "question": "看看身边一样东西，简单说说它是什么、放在哪儿。",
            "answer": "窗边有一盆绿色植物，放在小架子上，叶子挺精神。",
        },
        {
            "domain": "executive_function",
            "question": "如果下午下楼散步，您通常会先准备哪几样东西？",
            "answer": "我会先换鞋，带好门卡和纸巾，再看看外面热不热。",
        },
        {
            "domain": "attention",
            "question": "我们做个很短的数字练习，从 20 往回每次减 3，说三个数就好。",
            "answer": "20、17、14，我可以慢慢往下数。",
        },
        {
            "domain": "visuospatial",
            "question": "从客厅走到厨房，您通常会先经过哪儿？按家里的样子说就行。",
            "answer": "我会从客厅往餐桌旁边走，再转到厨房门口。",
        },
    ],
    "轻度下降": [
        {
            "domain": "orientation",
            "question": "{display_name}，您好，我是小顾。今天先从日期聊起，您觉得今天大概是周几？想到哪个就先说哪个。",
            "answer": "应该是周五吧，也可能是周六，我最近有点拿不准。",
        },
        {
            "domain": "memory",
            "question": "接下来记三个小东西：梨子、雨伞、公交卡。您先照着说一遍。",
            "answer": "梨子、雨伞……公交卡，我能跟着说，但怕等会儿会忘。",
        },
        {
            "domain": "language",
            "question": "选一个您看得到的东西，说说它长什么样、在什么地方。",
            "answer": "窗边有个绿色的盆栽，放在架子上，名字我一下想不起来。",
        },
        {
            "domain": "executive_function",
            "question": "如果下午想下楼走走，您一般会先做哪两件事？",
            "answer": "我会先换鞋，可能还要拿钥匙，其他要想一想。",
        },
        {
            "domain": "attention",
            "question": "我们试一个小数字，从 20 往回减 3，说三个就停。",
            "answer": "20、18、15……我知道要往回数，但中间有点乱。",
        },
        {
            "domain": "visuospatial",
            "question": "从客厅到厨房，中间通常会经过哪里？慢慢想，按顺序说。",
            "answer": "我大概先往餐桌那边走，再到厨房，但中间顺序有点想不清。",
        },
    ],
    "明显异常": [
        {
            "domain": "orientation",
            "question": "{display_name}，您好，我是小顾。我们从很简单的开始，您现在觉得是上午还是下午？",
            "answer": "我分不清，现在像早上又像下午，也不知道星期几。",
        },
        {
            "domain": "memory",
            "question": "这次只听三个词：梨子、雨伞、公交卡。记住哪个就说哪个。",
            "answer": "梨子……后面两个我没记住，你刚才说过吗？",
        },
        {
            "domain": "language",
            "question": "看一眼身边，告诉我一个您看到的东西就可以。",
            "answer": "那个……那个东西，我知道在那边，可说不出名字。",
        },
        {
            "domain": "executive_function",
            "question": "如果现在要出门，您会先做什么？说一个动作就可以。",
            "answer": "我不知道，要出门吗？我可能直接走，东西也想不起来。",
        },
        {
            "domain": "attention",
            "question": "我们换一个很短的数字。20 后面往回数一点点，可以吗？",
            "answer": "20、19、16……后面我乱了，不知道该怎么数。",
        },
        {
            "domain": "visuospatial",
            "question": "最后只问一个位置：厨房大概在客厅的哪边？",
            "answer": "我分不清哪边，厨房在哪里也说不好。",
        },
    ],
}
PRESET_INTERVIEW_QUESTIONS = [
    {
        "domain": "orientation",
        "question": "您好，我是小顾，今天陪您轻松聊几句。您知道今天大概是星期几吗？",
    },
    {
        "domain": "memory",
        "question": "我记下来了。接下来请您记三个词：梨子、雨伞、公交卡。先跟我说一遍可以吗？",
    },
    {
        "domain": "language",
        "question": "咱们换个轻松的话题。请您看看身边，任选一样东西，说说它是什么、在哪里。",
    },
    {
        "domain": "executive_function",
        "question": "我们聊聊平时习惯。如果下午要去楼下散步，您一般会先准备哪几样东西？",
    },
    {
        "domain": "attention",
        "question": "接下来做个短数字练习。请从 20 往回每次减 3，数三个数就可以。",
    },
    {
        "domain": "visuospatial",
        "question": "最后想请您想一想，从客厅走到厨房，通常会经过哪里？按您熟悉的路线说就好。",
    },
]
DIALOG_EXAMPLE_ANSWERS = {
    "orientation": {
        "normal": "今天是周六，我早上看了手机上的日期。",
        "mild_decline": "应该是周五吧，也可能是周六，我有点拿不准。",
        "vague": "我不太清楚，最近日子过得差不多。",
    },
    "memory": {
        "normal": "刚才聊到早饭，我吃了粥和鸡蛋。",
        "mild_decline": "我记得好像喝了粥，别的有点想不起来。",
        "vague": "这个我说不好，反正早上挺忙的。",
    },
    "language": {
        "normal": "窗边有一盆绿色植物，叶子看起来很精神。",
        "mild_decline": "窗边有个绿色的东西，具体叫什么我一时说不上来。",
        "vague": "屋里东西挺多的，我也不知道该说哪个。",
    },
    "executive_function": {
        "normal": "我会先换鞋，带好门卡和纸巾，再看看天气。",
        "mild_decline": "我会先换鞋吧，要带什么可能需要别人提醒一下。",
        "vague": "看情况吧，出去就出去，不一定要准备什么。",
    },
    "attention": {
        "normal": "20、17、14，我可以慢慢往下数。",
        "mild_decline": "20、18、15，后面我有点乱了。",
        "vague": "我不太想算这个，数字容易弄混。",
    },
    "visuospatial": {
        "normal": "我会从客厅往餐桌旁边走，再转到厨房门口。",
        "mild_decline": "我大概会往厨房那边走，中间经过哪里有点想不清。",
        "vague": "放哪边都差不多吧，我不太确定方向。",
    },
    "general": {
        "normal": "我能理解这个问题，可以按步骤回答。",
        "mild_decline": "我大概明白，但有些细节想不起来。",
        "vague": "我不太确定你问的是什么。",
    },
}

QUESTION_TYPE_KEYWORDS = {
    "orientation": ("星期", "日期", "今天", "时间", "几号"),
    "memory": ("刚才", "记得", "记忆", "记住", "三个词", "复述", "早饭", "早餐", "吃了什么"),
    "language": ("描述", "一句话", "看到", "东西", "房间"),
    "executive_function": ("准备", "出门", "计划", "步骤", "安排", "先"),
    "attention": ("往回数", "倒数", "数", "计算", "注意"),
    "visuospatial": ("左", "右", "方向", "位置", "路线", "经过", "客厅", "厨房", "门口"),
}
QUESTION_TYPE_INFER_ORDER = (
    "memory",
    "visuospatial",
    "attention",
    "executive_function",
    "language",
    "orientation",
)

QUESTION_SPECIFIC_EXAMPLES = {
    "season_preference": {
        "keywords": ("季节", "春天", "夏天", "秋天", "冬天"),
        "answers": {
            "normal": "我喜欢春天，天气暖和，花也开了。",
            "mild_decline": "我喜欢那个不冷不热的时候，具体叫什么我有点想不起来。",
            "vague": "都差不多吧，我也说不上来。",
        },
    },
    "digit_backward": {
        "keywords": ("倒着说", "倒背", "倒序", "这串数字"),
        "answers": {
            "normal": "1、8、3，我把它倒过来说。",
            "mild_decline": "1、3、8吧，我有点弄乱了。",
            "vague": "数字我记不住，你再说一遍吧。",
        },
    },
    "word_memory": {
        "keywords": ("请记住", "三个词", "词语", "稍后"),
        "answers": {
            "normal": "好的，我记住了：梨子、雨伞、公交卡。",
            "mild_decline": "我记得有梨子，后面两个词有点模糊。",
            "vague": "我不太确定刚才是哪几个词。",
        },
    },
    "visuospatial_position": {
        "keywords": ("左边", "右边", "位置", "旁边", "前面", "后面"),
        "answers": {
            "normal": "我能先看清参照物的位置，再说哪个在左边、哪个在右边。",
            "mild_decline": "左边右边我需要再看一下，不能马上确定。",
            "vague": "我不太分得清左右，可能要你指给我看。",
        },
    },
    "visuospatial_route": {
        "keywords": ("客厅", "厨房", "经过", "路线", "走到", "门口"),
        "answers": {
            "normal": "我通常会从客厅出来，经过餐桌旁边，再走到厨房门口。",
            "mild_decline": "我大概会先出客厅，再往厨房那边走，中间经过哪里有点记不清。",
            "vague": "我说不上具体路线，反正慢慢走过去就行。",
        },
    },
}
STALE_FIXED_ANSWER_MATERIALS = (
    ("苹果", "钥匙", "报纸"),
    ("钥匙", "水杯", "眼镜"),
)


def load_fixture_sessions(trajectory: str) -> list[dict[str, Any]]:
    filename = FIXTURE_FILES.get(trajectory)
    if filename is None:
        raise ValueError(f"Unknown demo trajectory: {trajectory}")

    path = FIXTURE_DIR / filename
    with path.open("r", encoding="utf-8") as file:
        sessions = json.load(file)

    return sessions


def load_all_fixture_sessions() -> dict[str, list[dict[str, Any]]]:
    return {
        trajectory: load_fixture_sessions(trajectory)
        for trajectory in FIXTURE_FILES
    }


def infer_dialog_question_type(question: str) -> str:
    clean_question = question.strip()
    for question_type in QUESTION_TYPE_INFER_ORDER:
        keywords = QUESTION_TYPE_KEYWORDS[question_type]
        if any(keyword in clean_question for keyword in keywords):
            return question_type
    return "general"


def normalize_dialog_domains(domains: Union[list[str], tuple[str, ...], set[str]]) -> list[str]:
    normalized: list[str] = []
    for domain in domains:
        if domain in COGNITIVE_DOMAINS and domain not in normalized:
            normalized.append(domain)
    return normalized


def get_covered_dialog_domains(turns: list[dict[str, str]]) -> list[str]:
    covered: list[str] = []
    for turn in turns:
        target_domain = turn.get("target_domain", "")
        domain = (
            target_domain
            if target_domain in COGNITIVE_DOMAINS
            else infer_dialog_question_type(turn.get("assistant", ""))
        )
        if domain in COGNITIVE_DOMAINS and domain not in covered:
            covered.append(domain)
    return covered


def get_next_preset_interview_question(
    covered_domains: Union[list[str], tuple[str, ...], set[str]],
) -> Optional[dict[str, str]]:
    covered = set(normalize_dialog_domains(covered_domains))
    for item in PRESET_INTERVIEW_QUESTIONS:
        if item["domain"] not in covered:
            return dict(item)
    return None


def all_dialog_domains_covered(
    covered_domains: Union[list[str], tuple[str, ...], set[str]],
) -> bool:
    return set(normalize_dialog_domains(covered_domains)) == set(COGNITIVE_DOMAINS)


def get_dialog_example_answers(
    question: str,
    target_domain: Optional[str] = None,
    sample_answers: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    specific = _specific_example_answers(question)
    if specific is not None:
        return specific

    candidate = _normalize_candidate_example_answers(
        question,
        target_domain=target_domain,
        sample_answers=sample_answers,
    )
    if candidate is not None:
        return candidate

    question_type = (
        target_domain
        if target_domain in DIALOG_EXAMPLE_ANSWERS
        else infer_dialog_question_type(question)
    )
    examples = DIALOG_EXAMPLE_ANSWERS.get(
        question_type,
        DIALOG_EXAMPLE_ANSWERS["general"],
    )
    return {answer_type: examples[answer_type] for answer_type in DIALOG_EXAMPLE_ANSWER_TYPES}


def build_classroom_demo_interview(
    cognitive_level: str,
    display_name: str = "张奶奶",
) -> list[dict[str, Any]]:
    level = cognitive_level if cognitive_level in CLASSROOM_DEMO_LEVEL_LABELS else "正常表现"
    expected_risk = CLASSROOM_DEMO_RISK_TENDENCY[level]
    trigger_plan = CLASSROOM_DEMO_CLOCK_TRIGGER_PLAN.get(level, {})
    trigger_after_turn = trigger_plan.get("after_turn")
    turns: list[dict[str, Any]] = []
    safe_display_name = str(display_name or "张奶奶").strip() or "张奶奶"

    dialogue_items = CLASSROOM_DEMO_DIALOGUES.get(level, CLASSROOM_DEMO_DIALOGUES["正常表现"])
    for turn_index, item in enumerate(dialogue_items, start=1):
        domain = item["domain"]
        question = str(item["question"]).format(display_name=safe_display_name)
        answer = item["answer"]
        clock_triggered = turn_index == trigger_after_turn
        turns.append(
            {
                "cognitive_level": level,
                "system_question": question,
                "patient_answer": answer,
                "target_domain": domain,
                "expected_risk": expected_risk,
                "is_simulated": True,
                "clock_triggered": clock_triggered,
                "clock_trigger_title": trigger_plan.get("title", "") if clock_triggered else "",
                "clock_trigger_elder_message": trigger_plan.get("elder_message", "") if clock_triggered else "",
                "clock_trigger_staff_note": trigger_plan.get("staff_message", "") if clock_triggered else "",
            }
        )

    return turns


def build_classroom_clock_report(cognitive_level: str, model: str = "") -> dict[str, Any]:
    level = cognitive_level if cognitive_level in CLASSROOM_DEMO_LEVEL_LABELS else "正常表现"
    profile = CLASSROOM_CLOCK_REPORT_PROFILES[level]
    domain_scores = {domain: None for domain in COGNITIVE_DOMAINS}
    domain_scores.update(profile["domain_scores"])
    findings = deepcopy(profile["clock_findings"])

    return {
        "session_id": f"classroom-clock-{_trajectory_key_for_classroom_level(level)}",
        "uploaded_filename": profile["filename"],
        "is_mock": True,
        "is_simulated": True,
        "classroom_demo_level": level,
        "domain_scores": domain_scores,
        "evidence": [
            {
                "domain": "visuospatial",
                "source": "clock",
                "text": text,
            }
            for text in findings.get("visuospatial_evidence", [])
        ],
        "clock_findings": findings,
        "cdt_features": deepcopy(profile["cdt_features"]),
        "risk_level": profile["risk_level"],
        "explanation": profile["explanation"],
        "disclaimer": DISCLAIMER,
        "metadata": {
            "source": "mock",
            "model": model.strip() or "classroom-clock-demo",
            "reason": "演示预置画钟结果",
        },
    }


def _trajectory_key_for_classroom_level(level: str) -> str:
    if level == "轻度下降":
        return "mild-decline"
    if level == "明显异常":
        return "obvious-issue"
    return "normal"


def _normalize_candidate_example_answers(
    question: str,
    target_domain: Optional[str],
    sample_answers: Optional[dict[str, Any]],
) -> Optional[dict[str, str]]:
    if not isinstance(sample_answers, dict):
        return None

    normalized: dict[str, str] = {}
    for answer_type in DIALOG_EXAMPLE_ANSWER_TYPES:
        value = sample_answers.get(answer_type)
        if not isinstance(value, str) or not value.strip():
            return None
        normalized[answer_type] = value.strip()

    if _candidate_answers_are_unrelated(question, target_domain, normalized):
        return None

    if _candidate_answers_look_like_stale_digit_task(question, normalized):
        return None

    if _answers_use_stale_fixed_material(normalized):
        return None

    return normalized


def _candidate_answers_are_unrelated(
    question: str,
    target_domain: Optional[str],
    answers: dict[str, str],
) -> bool:
    combined = " ".join(answers.values())
    inferred_domain = target_domain or infer_dialog_question_type(question)
    if inferred_domain == "visuospatial":
        spatial_keywords = (
            "左",
            "右",
            "位置",
            "旁边",
            "前面",
            "后面",
            "路线",
            "经过",
            "客厅",
            "厨房",
            "门口",
            "方向",
        )
        return not any(keyword in combined for keyword in spatial_keywords)

    if inferred_domain == "attention" and any(
        keyword in question for keyword in ("倒着", "倒背", "倒序", "这串数字")
    ):
        return True

    if inferred_domain == "memory" and any(
        keyword in question for keyword in ("请记住", "三个词", "稍后")
    ):
        return True

    return False


def _specific_example_answers(question: str) -> Optional[dict[str, str]]:
    clean_question = question.strip()
    if _question_matches_specific_type(clean_question, "digit_backward"):
        digit_answers = _digit_backward_example_answers(clean_question)
        if digit_answers is not None:
            return digit_answers

    if _question_matches_specific_type(clean_question, "word_memory"):
        word_answers = _word_memory_example_answers(clean_question)
        if word_answers is not None:
            return word_answers

    for item in QUESTION_SPECIFIC_EXAMPLES.values():
        if any(keyword in clean_question for keyword in item["keywords"]):
            answers = item["answers"]
            return {
                answer_type: answers[answer_type]
                for answer_type in DIALOG_EXAMPLE_ANSWER_TYPES
            }
    return None


def _answers_use_stale_fixed_material(answers: dict[str, str]) -> bool:
    combined = " ".join(answers.values())
    return any(
        all(material in combined for material in materials)
        for materials in STALE_FIXED_ANSWER_MATERIALS
    )


def _candidate_answers_look_like_stale_digit_task(
    question: str,
    answers: dict[str, str],
) -> bool:
    if any(keyword in question for keyword in ("倒着", "倒背", "倒序", "这串数字")):
        return False
    combined = " ".join(answers.values())
    if any(keyword in combined for keyword in ("倒着", "倒背", "倒序", "倒过来", "数字")):
        return True
    digit_like_answers = re.findall(r"(?:\d+[、,，\\-]){2,}\d+", combined)
    return bool(digit_like_answers)


def _question_matches_specific_type(question: str, example_type: str) -> bool:
    item = QUESTION_SPECIFIC_EXAMPLES.get(example_type)
    if not isinstance(item, dict):
        return False
    return any(keyword in question for keyword in item["keywords"])


def _digit_backward_example_answers(question: str) -> Optional[dict[str, str]]:
    digits = re.findall(r"\d+", question)
    if len(digits) < 2:
        return None

    reversed_digits = list(reversed(digits))
    normal = "-".join(reversed_digits)

    if len(reversed_digits) >= 3:
        confused_digits = [reversed_digits[0], reversed_digits[-1], *reversed_digits[1:-1]]
    else:
        confused_digits = digits

    return {
        "normal": f"{normal}。",
        "mild_decline": f"{'-'.join(confused_digits)}，我有点记混了。",
        "vague": "数字我记不住了，你再说一遍吧。",
    }


def _word_memory_example_answers(question: str) -> Optional[dict[str, str]]:
    words = _extract_memory_words(question)
    if len(words) < 2:
        return None

    joined_words = "、".join(words)
    first_word = words[0]
    remaining_count = max(len(words) - 1, 1)
    return {
        "normal": f"好的，我记住了：{joined_words}。",
        "mild_decline": f"我记得有{first_word}，后面{remaining_count}个词有点模糊。",
        "vague": "我不太确定刚才是哪几个词。",
    }


def _extract_memory_words(question: str) -> list[str]:
    candidate = question
    for separator in ("：", ":"):
        if separator in candidate:
            candidate = candidate.split(separator, 1)[1]
            break

    candidate = re.sub(r"[。！？!?].*$", "", candidate)
    parts = re.split(r"[、,，\s]+", candidate)
    words = [
        part.strip("。；;：:，,、 ")
        for part in parts
        if part.strip("。；;：:，,、 ")
    ]
    stop_words = {
        "请记住",
        "记住",
        "这三个词",
        "三个词",
        "稍后",
        "我会",
        "再问您",
    }
    return [word for word in words if word not in stop_words][:5]


def get_clock_sample_paths() -> dict[str, Path]:
    return {
        sample_key: CLOCK_SAMPLE_DIR / filename
        for sample_key, filename in CLOCK_SAMPLE_FILES.items()
    }
