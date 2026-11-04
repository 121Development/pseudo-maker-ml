"""Gazetteer of place names — fallback for cities the NER models miss.

Swedish municipality seats and larger towns, plus major international
cities common in CVs and contracts. Towns that are also ordinary Swedish
words or first names (Vara, Mora, Sala, Ale, Boden, Kil, Åre, Hjo, Nora …)
are left out on purpose; the models handle those in context.
"""
from __future__ import annotations

import re
from typing import Iterable

from .base import Span

SWEDISH_PLACES = """
Stockholm Göteborg Malmö Uppsala Västerås Örebro Linköping Helsingborg Jönköping Norrköping
Lund Umeå Gävle Borås Mölndal Södertälje Eskilstuna Karlstad Halmstad Växjö Sundsvall Luleå
Trollhättan Östersund Borlänge Falun Kalmar Skövde Kristianstad Karlskrona Skellefteå Uddevalla
Lidingö Motala Landskrona Örnsköldsvik Nyköping Karlskoga Varberg Trelleborg Lidköping Alingsås
Piteå Sandviken Ängelholm Solna Sollentuna Täby Nacka Huddinge Haninge Botkyrka Järfälla Upplands Väsby
Sigtuna Märsta Danderyd Vallentuna Österåker Åkersberga Värmdö Gustavsberg Tyresö Nynäshamn Norrtälje
Kista Bromma Vällingby Farsta Skärholmen Hägersten Solberga Sundbyberg Enköping Katrineholm Strängnäs
Kungsbacka Kungälv Partille Lerum Stenungsund Ulricehamn Falkenberg Laholm Båstad Höganäs Eslöv
Staffanstorp Lomma Vellinge Ystad Simrishamn Hässleholm Ronneby Karlshamn Sölvesborg Olofström
Västervik Oskarshamn Vimmerby Nässjö Värnamo Gislaved Vetlanda Tranås Ljungby Älmhult Lessebo
Mjölby Finspång Kumla Hallsberg Lindesberg Arboga Köping Fagersta Avesta Hedemora Ludvika Leksand
Rättvik Malung Sälen Hudiksvall Bollnäs Söderhamn Ljusdal Härnösand Sollefteå Kramfors Timrå
Kiruna Gällivare Kalix Haparanda Arvidsjaur Arjeplog Jokkmokk Lycksele Vilhelmina Storuman Hagfors
Arvika Kristinehamn Filipstad Säffle Åmål Mariestad Tidaholm Falköping Hofors Tierp Östhammar
Visby Gotland Öland Borgholm Kungshamn Lysekil Strömstad Tanum Orust Tjörn Öckerö Ale Härryda
Mölnlycke Kållered Hisingen Majorna Frölunda Angered Rosengård Limhamn Hyllie Sahlgrenska
""".split()
SWEDISH_PLACES = [p for p in SWEDISH_PLACES if p not in {"Ale"}] + ["Upplands Väsby"]

WORLD_CITIES = """
London Paris Berlin Hamburg Munich München Frankfurt Cologne Köln Zurich Zürich Geneva Genève Vienna Wien
Amsterdam Rotterdam Utrecht Brussels Bryssel Antwerp Luxembourg Copenhagen Köpenhamn Aarhus Odense Oslo Bergen
Trondheim Stavanger Helsinki Helsingfors Tampere Turku Åbo Reykjavik Tallinn Riga Vilnius Warsaw Warszawa
Krakow Kraków Gdansk Prague Prag Budapest Bucharest Sofia Athens Aten Rome Rom Milan Milano Turin Naples
Madrid Barcelona Valencia Seville Lisbon Lissabon Porto Dublin Manchester Liverpool Birmingham Bristol
Leeds Glasgow Edinburgh Cardiff Belfast Oxford Cambridge Newbury Lyon Marseille Toulouse Nice Bordeaux
Istanbul Ankara Moscow Moskva Kyiv Kiev Dubai Doha Riyadh Cairo Kairo Nairobi Lagos Johannesburg
Cape Town Mumbai Bangalore Delhi Beijing Peking Shanghai Shenzhen Tokyo Seoul Singapore Bangkok
Sydney Melbourne Auckland Toronto Montreal Vancouver Chicago Boston Seattle Austin Denver Houston
Atlanta Miami Dallas Washington Philadelphia San Francisco Los Angeles New York Mexico City
São Paulo Rio de Janeiro Buenos Aires Santiago Bogotá Lima Mogadishu Mogadishu Baghdad Bagdad
Damascus Damaskus Tehran Teheran Kabul Gothenburg
""".split()
WORLD_CITIES += ["Cape Town", "San Francisco", "Los Angeles", "New York", "Mexico City",
                 "São Paulo", "Rio de Janeiro", "Buenos Aires"]
_SINGLE_WORD_PARTS = {"Cape", "Town", "San", "Francisco", "Los", "Angeles", "New", "York", "Mexico",
                      "City", "São", "Paulo", "Rio", "de", "Janeiro", "Buenos", "Aires", "Upplands", "Väsby"}

_ALL = sorted({p for p in SWEDISH_PLACES + WORLD_CITIES if p not in _SINGLE_WORD_PARTS}, key=len, reverse=True)
_PATTERN = re.compile(r"(?<![\w@./-])(" + "|".join(re.escape(p) for p in _ALL) + r")(?![\wåäöÅÄÖ@-])")


class KnownPlaceRecognizer:
    name = "CITY"

    def find(self, text: str) -> Iterable[Span]:
        for m in _PATTERN.finditer(text):
            yield Span(m.start(), m.end(), self.name, m.group(0), 0.8, "pattern")
