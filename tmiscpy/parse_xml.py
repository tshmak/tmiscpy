from copy import copy
import xml.etree.ElementTree as ET

"""
A simple function to parse XML (SSML)
written with the help of copilot
"""

# Walk through elements
def parse_element(element, depth=0, attribs=dict()):
    result = []
    _attribs = copy(attribs)
    _attribs.update({element.tag: element.attrib})
    text = element.text.strip() if element.text else ''
    result0 = (text, _attribs)
    result.append(result0)
    if (element.tail or "").strip():
        result.append(
                (element.tail.strip(), attribs)
                )

    for child in element:
        _result = parse_element(child, depth + 1, _attribs)
        result.extend(_result) 

    return result

# Parse SSML
def parse_xml(xml): 
    root = ET.fromstring(xml)
    return parse_element(root)

if __name__ == '__main__': 
    # Test
    ssml = """
    <speak>
        Hello <break time="500ms"/> world!
        <jyutping> 
        <prosody rate="slow" pitch="+2st">This is SSML parsing.</prosody>
        This is jyutping text.
        </jyutping>
        This is normal text.
        <prosody rate="fast">This is SSML parsing number 2.</prosody>
        This is normal text 2.
        <prosody rate="fast">This is SSML parsing number 3.</prosody>
    </speak>
    """
    for line in parse_xml(ssml):
        print(line)
