import use

mod = use(
    "sqlalchemy",
    version="1.4.25",
    hashes={
        "33a1e86abad782e90976de36150d910748b58e02cd7d35680d441f9a76806c18",
        "a505ecc0642f52e7c65afb02cc6181377d833b7df0994ecde15943b18d0fa89c",
        "90fe429285b171bcc252e21515703bdc2a4721008d1f13aa5b7150336f8a8493",
        "1b38db2417b9f7005d6ceba7ce2a526bf10e3f6f635c0f163e6ed6a42b5b62b2",
        "a36ea43919e51b0de0c0bc52bcfdad7683f6ea9fb81b340cdabb9df0e045e0f7",
        "dd4ed12a775f2cde4519f4267d3601990a97d8ecde5c944ab06bfd6e8e8ea177",
        "7ad59e2e16578b6c1a2873e4888134112365605b08a6067dd91e899e026efa1c",
        "6400b22e4e41cc27623a9a75630b7719579cd9a3a2027bcf16ad5aaa9a7806c0",
        "2ed67aae8cde4d32aacbdba4f7f38183d14443b714498eada5e5a7a37769c0b7",
        "842c49dd584aedd75c2ee05f6c950730c3ffcddd21c5824ed0f820808387e1e3",
        "a79abdb404d9256afb8aeaa0d3a4bc7d3b6d8b66103d8b0f2f91febd3909976e",
        "9ebe49c3960aa2219292ea2e5df6acdc425fc828f2f3d50b4cfae1692bcb5f02",
        "e37621b37c73b034997b5116678862f38ee70e5a054821c7b19d0e55df270dec",
        "6003771ea597346ab1e97f2f58405c6cacbf6a308af3d28a9201a643c0ac7bb3",
        "a28fe28c359835f3be20c89efd517b35e8f97dbb2ca09c6cf0d9ac07f62d7ef6",
        "08d9396a2a38e672133266b31ed39b2b1f2b5ec712b5bff5e08033970563316a",
        "1ebd69365717becaa1b618220a3df97f7c08aa68e759491de516d1c3667bba54",
        "c211e8ec81522ce87b0b39f0cf0712c998d4305a030459a0e115a2b3dc71598f",
        "75cd5d48389a7635393ff5a9214b90695c06b3d74912109c3b00ce7392b69c6c",
        "6b602e3351f59f3999e9fb8b87e5b95cb2faab6a6ecdb482382ac6fdfbee5266",
        "91cd87d1de0111eaca11ccc3d31af441c753fa2bc22df72e5009cfb0a1af5b03",
        "e93978993a2ad0af43f132be3ea8805f56b2f2cd223403ec28d3e7d5c6d39ed1",
        "0b08a53e40b34205acfeb5328b832f44437956d673a6c09fce55c66ab0e54916",
        "16ef07e102d2d4f974ba9b0d4ac46345a411ad20ad988b3654d59ff08e553b1c",
        "26b0cd2d5c7ea96d3230cb20acac3d89de3b593339c1447b4d64bfcf4eac1110",
        "1adf3d25e2e33afbcd48cfad8076f9378793be43e7fec3e4334306cac6bec138",
        "41a916d815a3a23cb7fff8d11ad0c9b93369ac074e91e428075e088fe57d5358",
        "0566a6e90951590c0307c75f9176597c88ef4be2724958ca1d28e8ae05ec8822",
        "9a1df8c93a0dd9cef0839917f0c6c49f46c75810cf8852be49884da4a7de3c59",
        "7b7778a205f956755e05721eebf9f11a6ac18b2409bff5db53ce5fe7ede79831",
    },
    modes=use.auto_install,
)

print(mod.__name__)


def represent_num_as_base(num, base):
    """
    Represent a number in a different base.
    """
    if num == 0:
        return [0]
    digits = []
    while num:
        digits.append(num % base)
        num //= base
    return digits[::-1]
