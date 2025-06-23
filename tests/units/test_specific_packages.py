import sys
from types import ModuleType

import pytest

from justuse import auto_install, no_cleanup


@pytest.mark.parametrize(
    "package_name, module_name, version, hashes",
    (
        (
            "numpy",
            "numpy",
            "1.21.0",
            {
                "K孈諺搓稞㭯函攢訦冁舏在偦鸕覘延䅣微",
                "W余跒䌹㔟倠咥膆酐䄩物绁䣹萦䱒䀒嵲㫀",
                "U燊仓虑觖䢾䪍㼞䙼⑾鮹爠䂸泫驊鏓慥脲",
                "P䂹瀑蚜萃暧㡩栅訉慯忲磵殣怊伟躂岜㠽",
                "S驈烽堶堢苩觌宣陖阆㦯葅霙䇣锽鸋㻔摌",
                "鉳䥩猉罺䢢䚘㞐朤体颛軀辗﨏䍜圑蒷塶",
                "N䥐昺飂薏嵋㻝阫鎴蝧愩蚓臞敼鯩冓蝸銒",
                "M㐸丮䫈䩷ɽ燃䄻混蛳稝鉇役泃宵艞顃㼵",
                "l䭌㻧䈘樉䎜胵賡秖磗伳玌鄵㝷乺ʜ䩗㞄",
                "jǛ湄閈䌣鮴䥎堃㛓笸埩眣庚捡鈕㞻鍇錭",
                "O珌㡾㷗鴥拁嵹㒞㔹鐝Ȼ扡庇甄臞摉鮝厐",
                "蕧胸瓲工䛹侒囜第䞢棖頟䗯锩㠋㞭谶㥚",
                "P钉殯㓜魴壬谚蝇弄鈻馇蕔祏绿佡謊濔弡",
                "k蜆蓭鑶禬茌犿违維郈盇肀根棿嶠纭恦㗼",
                "Z序套䁳暙鐁鎨樫姡鳵曧頲㲿槤誐笷楆㖑",
                "J䘻鏤熆钣吚檭䟆颃稆㥌绊錽乸棈獒轧䐥",
                "k胨唃嫟鶛睉銤噶㝅婡䲍逪萻㲛撁佘籧趏",
                "k頃跊ó銋訋簴冴蛞铘诪韚秂坿櫹围㾶歆",
                "VƂ㫰雲巠䝲偯鷔餙䬸歏匧漃祌轘夽㑐㵿",
            },
        ),
    ),
)
def test_specific_packages(reuse, package_name, module_name, version, hashes):
    if sys.version_info < (3, 11):
        mod = reuse(
            (package_name, module_name),
            version=version,
            hashes=hashes,
            modes=auto_install | no_cleanup,
        )
        assert isinstance(mod, ModuleType)
