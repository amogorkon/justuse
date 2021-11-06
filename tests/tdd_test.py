import importlib
import os
import re
import shlex
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from shutil import rmtree
from unittest import skip

import pytest
import requests

from .unit_test import PyPI_Release, Version, log, reuse

not_local = "GITHUB_REF" in os.environ
is_win = sys.platform.lower().startswith("win")
not_win = not is_win

# Add in-progress tests here


def test_template(reuse):
    pass


def test_is_platform_compatible_macos(reuse):
    platform_tags = reuse.use.get_supported()
    platform_tag = next(iter(platform_tags))
    info = {
        "comment_text": "",
        "digests": {
            "md5": "2651049b70d2ec07d8afd7637f198807",
            "sha256": "cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff",
        },
        "downloads": -1,
        "filename": f"numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "has_sig": False,
        "md5_digest": "2651049b70d2ec07d8afd7637f198807",
        "packagetype": "bdist_wheel",
        "python_version": "source",
        "requires_python": ">=3.6",
        "size": 15599590,
        "upload_time": "2021-01-05T17:19:38",
        "upload_time_iso_8601": "2021-01-05T17:19:38.152665Z",
        "url": f"https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "yanked": False,
        "yanked_reason": None,
        "version": "1.19.5",
    }
    assert reuse._is_platform_compatible(PyPI_Release(**info), platform_tags)


def test_is_platform_compatible_win(reuse):
    platform_tags = reuse.use.get_supported()
    platform_tag = next(iter(platform_tags))
    info = {
        "comment_text": "",
        "digests": {
            "md5": "baf1bd7e3a8c19367103483d1fd61cfc",
            "sha256": "dbd18bcf4889b720ba13a27ec2f2aac1981bd41203b3a3b27ba7a33f88ae4827",
        },
        "downloads": -1,
        "filename": f"numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "has_sig": False,
        "md5_digest": "baf1bd7e3a8c19367103483d1fd61cfc",
        "packagetype": "bdist_wheel",
        "python_version": f"cp3{sys.version_info[1]}",
        "requires_python": f">=3.{sys.version_info[1]}",
        "size": 13227547,
        "upload_time": "2021-01-05T17:24:53",
        "upload_time_iso_8601": "2021-01-05T17:24:53.052845Z",
        "url": f"https://files.pythonhosted.org/packages/ea/bc/da526221bc111857c7ef39c3af670bbcf5e69c247b0d22e51986f6d0c5c2/numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "yanked": False,
        "yanked_reason": None,
        "version": "1.19.5",
    }
    assert reuse._is_platform_compatible(PyPI_Release(**info), platform_tags, include_sdist=False)


def test_pure_python_package(reuse):
    # https://pypi.org/project/example-pypi-package/
    file = reuse.Path.home() / ".justuse-python/packages/example_pypi_package-0.1.0-py3-none-any.whl"
    venv_dir = reuse.Path.home() / ".justuse-python/venv/example-pypi-package/0.1.0"
    file.unlink(missing_ok=True)
    if venv_dir.exists():
        rmtree(venv_dir)

    test = eval(
        b"reuse(\"example-pypi-package/examplepy\", version=\"0.1.0\", hashes={'S\xe3\xb5\x88\xe8\x9b\xb4\xe7\x9e\x99\xe7\xbb\xbd\xe3\xa1\x83\xe9\xb8\xa1 \xe3\x9c\x96\xe5\x83\xbc\xe6\xb1\xa0\xe6\xa2\xb5\xe9\xb0\xbc\xe4\xaf\x90\xe6\x9c\xa0\xe9\x89\x82\xe4\xa4\x9c\xe5\x9d\xa0\xe8\x91\x86', '7\xe6\x98\x93\xe6\x90\xbb\xe5\x80\x90\xe3\xba\x8d\xe4\x9a\xa1\xe5\x84\x99\xe7\x90\x9f\xe9\x99\xbb\xe3\xbf\xb7\xe5\x8c\xa6\xe4\x97\xaf\xe9\x97\x8d\xe8\x84\xb7\xe4\x87\x85\xe3\xab\x84\xe7\x83\xb6\xe5\x8c\x86'}, modes=reuse.auto_install)"
    )
    assert venv_dir.exists() == False, "Should not have created venv for example-pypi-package"

    assert str(test.Number(2)) == "2"
    if file.exists():
        file.unlink()


def test_db_setup(reuse):
    assert reuse.registry


def installed_or_skip(reuse, name, version=None):
    if not (spec := find_spec(name)):
        pytest.skip(f"{name} not installed")
        return False
    try:
        dist = distribution(spec.name)
    except PackageNotFoundError as pnfe:
        pytest.skip(f"{name} partially installed: {spec=}, {pnfe}")

    if not (
        (ver := dist.metadata["version"])
        and (not version or reuse.Version(version)) == (not ver or reuse.Version(ver))
    ):
        pytest.skip(f"found '{name}' v{ver}, but require v{version}")
        return False
    return True


@pytest.mark.skipif(not_local, reason="requires matplotlib")
def test_use_str(reuse):
    if not installed_or_skip(reuse, "matplotlib"):
        return
    mod = reuse("matplotlib/matplotlib.pyplot")
    assert mod


@pytest.mark.skipif(not_local, reason="requires matplotlib")
def test_use_tuple(reuse):
    if not installed_or_skip(reuse, "matplotlib"):
        return
    mod = reuse(("matplotlib", "matplotlib.pyplot"))
    assert mod


@pytest.mark.skipif(not_local, reason="requires matplotlib")
def test_use_kwargs(reuse):
    if not installed_or_skip(reuse, "matplotlib"):
        return
    mod = reuse(package_name="matplotlib", module_name="matplotlib.pyplot")
    assert mod


def test_auto_install_protobuf(reuse):
    mod = reuse(
        "protobuf",
        version="3.19.1",
        hashes={
            "4葢鞮陕戛戛柈鳢窄濝玣妺盙㜻蜺㙎䴑紨",
            "E熅䚓隕笊䅏燕㯊蕄叄伵搖坪詵詾韋熜䒦",
            "C肵悽虪䯧齒蔘綠裺敬挥悈訰敤媨祒栬呻",
            "D儊 䈫多㐓蒞㣋燸販䚨迶瑅辍芆朢袟媜䄧",
            "B烊崯浴缮泴䟇还珬㒽䶂路㐑䵖腺镙羧灓",
            "F曅井垩踲涜䴜䓧䐈頉貱猯㢛㢔锊巂喇洕",
            "T樦桙䏝鶞躸螮鴢診塆檎䥲㱐䮾踉䴡歅㜁",
            "C钇樤䏽篗㔚箦鶅駆檅浲名薵葰儴憻弴甊",
            "構鵲学蒆肊膕荶漎鐳䘏澦顪櫯倽輇狫氇",
            "I甓肅瘿嬥㼈鈈㸮 夿儬聙䰭䀴㝦啈䦧裬㴇",
            "T鉉舀莨鏀䓨厳酆艧䋖荳㦵雝楛瑏䶌飢㔿",
            "M倱㺣酻㹑毚䫣儴慾歰蜩噷柠菖斻麖䜷䔐",
            "V縎䤓膋䂏嗑㢞协㿄鼿务黏檇藠穦㕕癦掆",
            "F鉺绀缩萼歞㾶疥烃垦孟潞籕詞屵摿著迟",
            "V箂䠫仹䬫㟏侺檺㘂蓓厴㟡綝颾䪳盨旽詨",
            "Y箸錇曪艻庤褏疴䋇糈载同吵翦嘚錩鎬煘",
            "7烓罇埞鐊謜烗㡇耶蕝晑澽鏰輚㼏靣鉥閪",
            "6舒櫚餗䨶⒐㿚萚鈀字㳏惏䵐颲辖䡽 㜿蚁",
            "J㡠鑶瓬勪㵦佧褗䗄㕋腮㑪埤傝姱畉轒绺",
            "V硄頎僊蛄僥幌鹁踖燌約忿鷤逪雁㤱讽",
            "T攱俭銝奞掲鬩淊錙鸱㤷楑鯚味鲒萠几魌",
            "O胵歉杭鐏麛秽媄旛屄葝笊䦢郎睌㟋槺狲",
            "T扬减④塏訟潣枓噟朤鮿懠欚䯩惗蛤毭䡫",
            "F㓞犠璫蒕䄩㱉笘晟梮抎雱钄癦争荋跸碊",
        },
        modes=reuse.auto_install,
    )
    assert mod


def test_auto_install_numpy(reuse):
    mod = reuse(
        "numpy",
        version="1.21.3",
        hashes={
            "J嶖㲷羀砿惏辕䞜蝙碅洆敋郟藓䥏鶫険悹",
            "瑍 呯誏聦胛㕥䘓䉦䟿辺贼⒔貢酨骇畜药",
            "U佯賁荍㲕鬝㰞崛籷䨸㧓䫫垮翗恕㭐跕轤",
            "7攀鮮蠼 蝮訁膭刁䕘㮌㴳䮔講䢶贏颌䊚查",
            "4蚌珙㑍鐋敮䅀䗍櫁嶆䄘谍䬋㣬㳲婛胖檥",
            "I揶盝䕴誉祫檹塑媝邝簋啻气枑暐㟺䋎獞",
            "I乩苟插㴂萀繩兮䛧螩俜轗躵缑䎩⑴汣䛮",
            "4陘郶䋠蝺涸蔼悕 婇縯㾉㖺慺壥雍仜赝畄",
            "U䆌㞏辩嵉攺鞘赅痊瓖潣霽麚䗧糷澆迵鄨",
            "2摱餜梮㕅平䆅䜺磪齰鎶昏椆笉燴艊弫蠤",
            "Q齀窷嘥頺諻㓫㑍璢㞂魥遶愃䨺浑唒槞疼",
            "F疄碶懍崁䍎瀔薨销絷鷢噆骤㕡焲玄砛葭",
            "4鴸㞩㰲児䏻蔯䄒䦆䤼謤絵㻮脏訬惲㤏㮏",
            "K挂爏籹鼺郖竻䛖㮤䒓墳贇䐶渗鴅嘂龢騼",
            "Y豷壘餝䘮枳慖傉鹑繁䜨䥿耚韨嘴鰾褋䇴",
            "A壴犡䍕屛籢噪䑴浤芡斎吤壽㺩㿙瑽倛䤪",
            "A峡鲻倠魣垚㽙詾寜鬉䬂泃籭㤜珁翹喘瑯",
            "C黡㦓倹醶樜赙仙诖蠴䜾鋚霃㡡橾踗怺",
            "5蘗䡁蘆鈾㧿妋簕礓䯁蝌㛣鉶駤滛騔蝽釛",
            "9巻組䟵㗫艛䬙襨㱿音蠙辺䬸棑鍂㳉㱧峲",
            "V廠赒歾垇㪢騋衮梬壕楒卉鼆湛蜂㿫䨼閗",
            "R萱虹䓧窔餿䍷乁抶㶁遁坺苔苘粜仗䴟鱧",
            "2㻢䆸罆㓼曬窋郮鸴慍鵍礴葑觔䎢䩋銞瘡",
            "Q㟊蓣䳺洕韸䴞頋䇎䤪䐞捗濪飩鮮䐆㱰互",
            "X崊㞅豄縑葓↑烌䵏楳袲氊蟛霝㻳灍㖨陁",
            "M鎵触珓趆欁M汅䰎鏍䜳仦釈箛䱁畑鱭贶",
            "6㰐輥藹䘓樚傃㕴㧓餪掗蟷顎裌伷䏆撿胫",
            "V厶繾鍈幂鐸徘樻屏繙㹊馶㜥雟樨㹶緤咮",
            "C䵊僊晚㨦曰䁢有圱樊聨巓⒘誻錤凉睼亶",
            "F浛䀬桗䶭㕥碂埄覓喓笽勻欓㲞㢏粝㓈稹",
            "Y㪅猐巭窆㕛嗭枬孞籿禋套 䉛㶭袜颙琪䇧",
            "B鼝閧芇㑊炽憀芷䶟轉谆櫐睷耴䁳郒卻丯",
            "9诒㣯軖癙鴎阜特靔濅嵴皖鄳湴靫虐餼䭌",
        },
        modes=reuse.auto_install,
    )
    assert mod


def test_auto_install_justuse(reuse):
    mod = reuse("justuse/use", version="0.5.0", hashes={"9瑉㬾鮓蘄鏲枲亝㫖徑怌罒硴厷䣍拑瑙瓦"}, modes=reuse.auto_install)
    assert mod
