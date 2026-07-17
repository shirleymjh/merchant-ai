from merchant_ai.config import get_settings
from merchant_ai.services.assets import (
    HybridRecallService,
    TopicAssetService,
    split_markdown_for_recall,
)


def test_rule_document_chunks_follow_heading_hierarchy_and_paragraphs():
    markdown = """# 平台规则

## 退款规则

适用条件：订单已经支付。
处理结论：进入退款审核。

例外情况：风险订单需要人工复核。

## 发货规则

商家需要在承诺时间内发货。
"""

    chunks = split_markdown_for_recall(markdown, "rule", target_chars=80, max_chars=140, overlap_chars=20)

    assert chunks
    assert chunks[0]["headingPath"] == ["平台规则", "退款规则"]
    assert "适用条件" in chunks[0]["content"]
    assert "处理结论" in chunks[0]["content"]
    assert any(chunk["headingPath"] == ["平台规则", "发货规则"] for chunk in chunks)
    assert all(len(chunk["content"]) <= 140 for chunk in chunks)


def test_long_section_uses_small_overlap_without_crossing_heading_boundary():
    markdown = "# 规则\n\n## 条款 A\n\n%s\n\n%s\n\n## 条款 B\n\n独立条款。" % (
        "第一段内容。" * 25,
        "第二段内容。" * 25,
    )

    chunks = split_markdown_for_recall(markdown, "rule", target_chars=120, max_chars=180, overlap_chars=24)
    section_a = [chunk for chunk in chunks if chunk["headingText"].endswith("条款 A")]
    section_b = [chunk for chunk in chunks if chunk["headingText"].endswith("条款 B")]

    assert len(section_a) > 1
    assert any(chunk["overlapChars"] > 0 for chunk in section_a[1:])
    assert section_b[0]["overlapChars"] == 0
    assert "条款 A" not in section_b[0]["content"]


def test_hybrid_recall_builds_one_rule_document_per_markdown_chunk(tmp_path):
    rules = tmp_path / "rules"
    topics = tmp_path / "topics"
    rules.mkdir()
    topics.mkdir()
    (rules / "rule.md").write_text(
        "# 平台规则\n\n## 退款\n\n退款处理规则。\n\n## 发货\n\n发货处理规则。",
        encoding="utf-8",
    )
    settings = get_settings().model_copy(
        update={
            "rule_knowledge_path": str(rules),
            "topic_path": str(topics),
            "rule_chunk_target_chars": 80,
            "rule_chunk_max_chars": 140,
            "rule_chunk_overlap_chars": 10,
        }
    )
    topic_assets = TopicAssetService(settings)
    recall = HybridRecallService(settings, topic_assets)

    docs = recall._load_documents()

    assert len(docs) == 2
    assert all(doc.source_type == "GOVERNED_RULE" for doc in docs)
    assert docs[0].metadata["chunkStrategy"] == "langchain_markdown_header_recursive"
    assert docs[0].doc_id == "semantic:rules:rule:chunk:0000"
    assert docs[0].metadata["sourcePath"] == "rules/rule.md"
    assert docs[1].metadata["headingPath"] == ["平台规则", "发货"]
