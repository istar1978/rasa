import pytest
from rasa.dummy import *

def test_one():
    with pytest.warns(UserWarning) as record:
        print_a_warning()
    assert len(record) == 1
    assert (
        "A useful message" in record[0].message.args[0]
    )

def test_two():
    with pytest.warns(UserWarning) as record:
        print_two_warnings()
    assert len(record) == 2
    assert (
        "Numero" in record[0].message.args[0]
    )
    assert (
        "Another one" in record[1].message.args[0]
    )

def test_zero():
    with pytest.warns(None) as record:
        you_shall_not()
    assert len(record) == 0

def test_bad():
    print_a_warning()
