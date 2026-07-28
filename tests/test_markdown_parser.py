from capsule.parsers.markdown import MarkdownParser


def test_images_in_group_inherit_preceding_heading() -> None:
    source = """
# 午后-黄昏

![](images/one.png)
![](images/two.png)
"""

    result = MarkdownParser().parse(source)

    assert len(result.image_references) == 2
    for reference in result.image_references:
        assert reference.contexts[0].text == "午后-黄昏"
        assert reference.contexts[0].relation_type == "preceding_heading"


def test_image_caption_precedes_document_context() -> None:
    source = """
## 角色设定

![银发角色](images/character.png)
"""

    result = MarkdownParser().parse(source)
    contexts = result.image_references[0].contexts

    assert [context.relation_type for context in contexts] == [
        "caption",
        "preceding_heading",
    ]
    assert [context.text for context in contexts] == ["银发角色", "角色设定"]
