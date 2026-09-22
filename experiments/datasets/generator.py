"""Dataset generator for the Pre-Module-5 Research Validation Experiment.

Constructs 120 clean, natural factual claims across 4 domains (science, history, technology, news),
balanced 15 SUPPORTED and 15 CONTRADICTED per domain (60 SUPPORTED, 60 CONTRADICTED total).
Each claim includes 5 distinct independent root evidence passages.
Generates controlled ground-truth provenance annotations.
"""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import Any

# Seed for complete determinism (§3.1)
DEFAULT_SEED = 42

DOMAINS = ["science", "history", "technology", "news"]

# 120 Curated factual claims with ground-truth labels and root factual narratives
# Exactly 30 per domain: 15 SUPPORTED, 15 CONTRADICTED
RAW_CLAIMS_DATA: list[tuple[str, str, str, tuple[str, str, str, str, str]]] = [
    # =========================================================================
    # SCIENCE (30 claims: 15 SUPPORTED, 15 CONTRADICTED)
    # =========================================================================
    (
        "The James Webb Space Telescope operates primarily in infrared wavelengths.",
        "science", "SUPPORTED",
        (
            "The James Webb Space Telescope (JWST) is an infrared space observatory launched on December 25, 2021. Designed to succeed Hubble, it features high-resolution instruments sensitive from long-wavelength visible light through mid-infrared.",
            "JWST's scientific payload comprises four instruments: NIRCam, NIRSpec, MIRI, and FGS/NIRISS. All four instruments operate between 0.6 and 28.3 microns in the infrared spectrum.",
            "Unlike the Hubble Space Telescope which focuses largely on optical and ultraviolet bands, Webb's primary mirror is specifically engineered to collect infrared light.",
            "Astronomical observations by JWST rely on infrared detectors to pierce obscuring cosmic dust and examine early star-forming galaxies redshifted into infrared wavelengths.",
            "The space observatory's focal plane arrays utilize mercury-cadmium-telluride sensors optimized for near-infrared detection and arsenic-doped silicon sensors for mid-infrared astronomy.",
        ),
    ),
    (
        "Photosynthesis in plants converts water and carbon dioxide into glucose and oxygen.",
        "science", "SUPPORTED",
        (
            "Oxygenic photosynthesis is the biochemical process used by green plants to convert light energy into chemical energy, converting water and carbon dioxide into glucose while releasing oxygen.",
            "The light-dependent reactions of photosynthesis split water molecules to produce oxygen gas, ATP, and NADPH in the chloroplast thylakoids.",
            "During the Calvin cycle, enzymes utilize carbon dioxide along with ATP and NADPH to synthesize six-carbon glucose sugars in plant chloroplasts.",
            "Plant biochemical pathways require water, carbon dioxide, and sunlight to carry out photosynthesis, yielding glucose for cellular energy and releasing oxygen into the atmosphere.",
            "Chloroplasts contain chlorophyll pigments that absorb light energy to drive the synthesis of glucose from carbon dioxide and water with oxygen byproduct.",
        ),
    ),
    (
        "Nuclear fusion in the core of the Sun converts helium into uranium.",
        "science", "CONTRADICTED",
        (
            "The core of the Sun generates energy through the proton-proton chain reaction, a nuclear fusion process that converts hydrogen nuclei into helium-4, not uranium.",
            "Current solar models confirm that the Sun's core does not fuse helium into uranium. The Sun fuses hydrogen into helium at 15 million Kelvin.",
            "Uranium synthesis requires rapid neutron capture occurring in supernova explosions and neutron star mergers, never in main-sequence dwarf stars like the Sun.",
            "Astrophysical observations of solar neutrinos confirm that hydrogen-to-helium fusion powers the Sun. The Sun lacks the temperature and pressure to synthesize uranium.",
            "The Sun converts approximately 600 million tons of hydrogen into helium every second in its core. Main-sequence stars do not produce uranium in their cores.",
        ),
    ),
    (
        "Mitochondria generate the majority of cellular adenosine triphosphate (ATP).",
        "science", "SUPPORTED",
        (
            "Mitochondria are double-membrane-bound cellular organelles that produce over ninety percent of cellular adenosine triphosphate (ATP) through oxidative phosphorylation.",
            "The inner mitochondrial membrane houses the electron transport chain complexes that generate ATP from ADP during aerobic cellular respiration.",
            "Known as the powerhouses of the eukaryotic cell, mitochondria utilize proton gradients across their cristae to drive ATP synthase and produce ATP.",
            "Cellular biochemistry confirms that mitochondrial aerobic respiration yields significantly more ATP molecules per glucose molecule than cytoplasmic glycolysis alone.",
            "Mitochondrial enzymes metabolize pyruvate and fatty acids via the citric acid cycle to generate reducing equivalents for massive ATP synthesis.",
        ),
    ),
    (
        "DNA double helix structure consists of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
        "science", "SUPPORTED",
        (
            "Deoxyribonucleic acid (DNA) is composed of two polynucleotide chains forming a double helix, with base pairing between adenine, thymine, guanine, and cytosine.",
            "Watson and Crick discovered that the DNA double helix is stabilized by hydrogen bonds between complementary base pairs: adenine with thymine, and guanine with cytosine.",
            "The genetic alphabet of genomic DNA in all cellular life is encoded by four nitrogenous nucleotide bases: adenine (A), cytosine (C), guanine (G), and thymine (T).",
            "Molecular biology demonstrates that DNA replication preserves genetic information through complementary pairing of adenine with thymine and cytosine with guanine.",
            "Chromosomal DNA polymer chains are constructed from deoxyribose sugars, phosphate groups, and the four canonical nitrogenous bases adenine, thymine, guanine, and cytosine.",
        ),
    ),
    (
        "Sound waves travel faster through a vacuum than through liquid water.",
        "science", "CONTRADICTED",
        (
            "Sound waves are mechanical vibrations that require a material medium to propagate and cannot travel through a vacuum at all.",
            "In liquid water, sound propagates at approximately 1,480 meters per second, whereas in a vacuum the speed of sound is exactly zero because no matter exists to transmit acoustic pressure.",
            "Acoustic physics establishes that mechanical sound waves cannot travel across a vacuum, making sound transmission impossible in a vacuum.",
            "Experiments confirm that sound waves fail to propagate in an evacuated chamber, whereas sound travels rapidly through water and other dense fluid media.",
            "Because sound requires intermolecular collisions to transmit acoustic energy, sound travels at roughly 1,500 m/s in water but cannot propagate in a vacuum.",
        ),
    ),
    (
        "The human skeleton in an adult typically consists of 206 distinct bones.",
        "science", "SUPPORTED",
        (
            "The adult human skeleton typically comprises 206 individual bones organized into axial and appendicular divisions.",
            "Medical anatomy textbooks classify the 206 bones of the mature human skeletal system, including the cranial bones, vertebrae, ribs, and limb bones.",
            "While infants are born with approximately 270 bones, bone fusion during postnatal development results in a standard total of 206 bones in the adult human skeleton.",
            "Radiological and anatomical consensus establishes 206 as the standard count of articulated bones in an adult human body.",
            "The skeletal framework of a healthy adult human contains 206 bones, providing structural support and protecting vital internal organs.",
        ),
    ),
    (
        "Pure liquid water at standard sea-level atmospheric pressure boils at 50 degrees Celsius.",
        "science", "CONTRADICTED",
        (
            "Under standard sea-level atmospheric pressure of 1 atmosphere (101.3 kPa), pure water boils at 100 degrees Celsius, not 50 degrees Celsius.",
            "Thermodynamic tables establish the boiling point of pure liquid water as 100 °C (212 °F) at standard sea-level barometric pressure.",
            "Laboratory measurement confirms that liquid water requires heating to 100 °C at 1 atm to reach vapor pressure equilibrium and undergo boiling.",
            "At 50 degrees Celsius under standard atmospheric pressure, liquid water has a vapor pressure of only 12.3 kPa, which is far below atmospheric pressure, preventing boiling.",
            "Physical chemistry defines the normal boiling point of pure water as 100 degrees Celsius at standard atmospheric pressure of 760 mmHg.",
        ),
    ),
    (
        "Electrons possess a positive electric charge, while protons carry a negative charge.",
        "science", "CONTRADICTED",
        (
            "In atomic physics, electrons are elementary leptons carrying an electric charge of negative one elementary charge (-1e), while protons carry positive one (+1e).",
            "The Standard Model of particle physics defines electrons as having a negative charge of -1.602 x 10^-19 Coulombs, whereas protons have an equal but positive charge.",
            "Subatomic particles in atomic structure consist of positively charged nuclear protons and negatively charged extranuclear electrons orbiting the nucleus.",
            "Electromagnetic theory establishes that protons have a positive electrical charge and electrons carry a negative charge, creating attractive Coulomb forces in atoms.",
            "Experimental physics confirms that an electron has a negative fundamental charge, whereas a proton has an equal magnitude positive fundamental charge.",
        ),
    ),
    (
        "The speed of light in a vacuum is approximately 100 kilometers per hour.",
        "science", "CONTRADICTED",
        (
            "The speed of light in vacuum, denoted c, is a universal physical constant precisely defined as 299,792,458 meters per second, or approximately 300,000 km/s.",
            "Special relativity establishes that electromagnetic radiation travels through a vacuum at the invariant speed of roughly 300,000 kilometers per second.",
            "Experimental measurements of light propagation in vacuum confirm a constant speed of approximately 3 x 10^8 meters per second (300,000 km/s).",
            "Optical physics defines the constant c as exactly 299,792.458 km/s in vacuum, which rounds to 300,000 kilometers per second in standard usage.",
            "Photons in a vacuum propagate at the universal cosmic speed limit of roughly 300,000 km per second, as established by modern relativistic physics.",
        ),
    ),
    (
        "The Earth is located at the center of the solar system with all planets orbiting it.",
        "science", "CONTRADICTED",
        (
            "Modern astronomical science proved that the Sun is at the center of the solar system, with Earth and other planets orbiting the Sun in heliocentric orbits.",
            "The geocentric model claiming Earth sits at the center was definitively refuted by Copernicus, Galileo, and Kepler in favor of the heliocentric solar system.",
            "Gravitational dynamics demonstrate that Earth and other planets revolve around the massive central Sun, contradicting ancient geocentric claims.",
            "Planetary orbital observations confirm that Earth is the third planet orbiting the central Sun, not the center of the solar system.",
            "Space exploration and telescopic telemetry prove that the Sun is the gravitational center of our solar system, with Earth orbiting it at 1 AU.",
        ),
    ),
    (
        "Mercury is the largest planet by diameter and mass in the solar system.",
        "science", "CONTRADICTED",
        (
            "Jupiter is the largest planet in our solar system, with a diameter of about 142,984 kilometers and more than twice the mass of all other planets combined.",
            "Planetary astronomy confirms that Jupiter is the most massive planet in the solar system, possessing an equatorial radius exceeding 71,000 kilometers.",
            "NASA planetary surveys confirm Jupiter as the largest celestial body in the solar system aside from the Sun, with 318 times the mass of Earth.",
            "Observations by space probes confirm that Jupiter's volume could hold over 1,300 Earths, making it unequivocally the largest planet in our solar system.",
            "Jupiter's enormous mass and diameter rank it as the premier gas giant and largest planet orbiting the Sun.",
        ),
    ),
    (
        "Diamonds and graphite are both allotropes composed entirely of carbon atoms.",
        "science", "SUPPORTED",
        (
            "Diamond and graphite are two well-known allotropes of the chemical element carbon, consisting entirely of carbon atoms arranged in different crystal lattices.",
            "Chemical analysis confirms that pure diamond and pure graphite contain only carbon atoms, differing solely in their covalent bonding geometry.",
            "In diamond, carbon atoms form tetrahedral sp3 lattices, while in graphite they form planar hexagonal sp2 sheets, both being pure carbon allotropes.",
            "Materials science classifies both diamond and graphite as allotropic forms of elemental carbon, sharing identical elemental composition.",
            "Combustion of either pure diamond or pure graphite in pure oxygen yields exclusively carbon dioxide, proving both are pure carbon.",
        ),
    ),
    (
        "Humans have three biological lungs located inside the thoracic cavity.",
        "science", "CONTRADICTED",
        (
            "Human anatomy establishes that humans have exactly two biological lungs, a right lung and a left lung, located in the thoracic cavity.",
            "Standard human pulmonary anatomy comprises two lungs: the right lung with three lobes and the left lung with two lobes, not three separate lungs.",
            "Medical examinations confirm that healthy humans possess a pair of lungs (two lungs), which flank the heart in the chest cavity.",
            "Surgical textbooks identify two lungs in the human respiratory system, refuting the notion that humans have three distinct lungs.",
            "Anatomical dissection and medical imaging show that normal human respiratory anatomy consists of two lungs connected to the trachea.",
        ),
    ),
    (
        "Penicillin was the first broadly effective modern antibiotic discovered by Alexander Fleming.",
        "science", "SUPPORTED",
        (
            "Scottish bacteriologist Alexander Fleming discovered the antibiotic substance penicillin produced by Penicillium notatum mold at St. Mary's Hospital in 1928.",
            "Fleming's 1928 discovery of penicillin revolutionized medicine by introducing the world's first widely effective antibacterial antibiotic.",
            "Medical history credits Alexander Fleming with discovering penicillin in September 1928 when mold contaminated a Staphylococcus culture plate.",
            "The development of penicillin as a clinical antibiotic followed Fleming's initial 1928 laboratory discovery of the antibacterial mold secretion.",
            "Penicillin was isolated and described by Alexander Fleming in 1928, inaugurating the modern era of antibiotic therapeutics.",
        ),
    ),
    (
        "The Moon possesses an atmosphere denser than that of the Earth.",
        "science", "CONTRADICTED",
        (
            "The Moon has an extremely tenuous surface boundary exosphere with a pressure of about 3 x 10^-15 atm, which is virtually a vacuum compared to Earth's atmosphere.",
            "Lunar atmospheric density is more than fourteen orders of magnitude lower than Earth's sea-level atmospheric density.",
            "Apollo surface experiments proved that the Moon has virtually no atmosphere, possessing only trace gases that do not form a dense atmospheric envelope.",
            "Astrophysical measurements confirm that the Moon lacks a substantial atmosphere, making Earth's atmosphere vastly denser by comparison.",
            "Because of its low gravity and lack of a global magnetic field, the Moon cannot retain a dense atmosphere, unlike Earth.",
        ),
    ),
    (
        "Helium is a noble gas with the atomic number 2 on the periodic table.",
        "science", "SUPPORTED",
        (
            "Helium is the second element on the periodic table, possessing an atomic number of 2 and belonging to Group 18 noble gases.",
            "The periodic table places helium in Group 18 with atomic number 2, characterized by a complete valence shell of two electrons.",
            "In chemistry, helium has chemical symbol He, atomic number 2, and is classified as an inert noble gas under standard conditions.",
            "Nuclear physics confirms that a helium nucleus consists of two protons (atomic number 2), placing it as the lightest noble gas.",
            "Helium is positioned at the top of the noble gas column with atomic number 2, having an unreactive closed 1s2 electron shell.",
        ),
    ),
    (
        "Plate tectonics explains that Earth's lithosphere is divided into moving rigid plates.",
        "science", "SUPPORTED",
        (
            "The geological theory of plate tectonics states that Earth's outer lithospheric shell is divided into several major and minor rigid tectonic plates.",
            "Geoscientific consensus confirms that Earth's rigid lithosphere floats on the ductile asthenosphere, moving relative to one another at plate boundaries.",
            "Plate tectonics describes how continental and oceanic plates move across Earth's mantle, driving continental drift, earthquakes, and volcanism.",
            "Modern geology establishes that Earth's surface comprises moving lithospheric plates whose interactions generate mountain ranges and ocean trenches.",
            "Seismological and geological evidence validates plate tectonics as the unifying model explaining lithospheric plate motions across the globe.",
        ),
    ),
    (
        "Venus has a surface temperature colder than that of Mars.",
        "science", "CONTRADICTED",
        (
            "Venus has an average surface temperature of approximately 465 degrees Celsius, making it the hottest planet in the solar system, far hotter than Mars.",
            "Mars has an average surface temperature of minus 60 degrees Celsius, whereas Venus suffers from a runaway greenhouse effect reaching over 450 °C.",
            "Planetary surveys show that Venus's dense carbon dioxide atmosphere traps intense heat, producing surface temperatures much higher than cold Martian surfaces.",
            "Space probe data from Magellan and Viking confirm that Venus is drastically hotter than Mars, refuting any claim that Venus is colder.",
            "The extreme atmospheric greenhouse effect makes Venus's surface over 500 degrees hotter than Mars's frigid sub-zero environment.",
        ),
    ),
    (
        "Enzymes function as biological catalysts that accelerate chemical reactions in living cells.",
        "science", "SUPPORTED",
        (
            "Enzymes are macromolecular biological catalysts, primarily proteins, that dramatically increase the rate of biochemical reactions by lowering activation energy.",
            "Biochemical research establishes that enzymes accelerate specific metabolic reactions in living organisms without being consumed in the process.",
            "In cellular metabolism, enzymes bind substrate molecules at their active sites to catalyze vital chemical transformations at physiological temperatures.",
            "Molecular biology defines enzymes as catalytic agents that accelerate cellular biochemical reactions by factors of up to millions of times.",
            "Enzymatic catalysis is essential for all living cellular processes, allowing metabolic pathways to proceed rapidly and selectively.",
        ),
    ),
    (
        "The atomic number of an element is determined solely by the number of neutrons in its nucleus.",
        "science", "CONTRADICTED",
        (
            "In chemistry and atomic physics, the atomic number of an element is determined exclusively by the number of protons in its nucleus, not neutrons.",
            "The number of neutrons defines an atom's isotope, whereas the atomic number (Z) is defined strictly as the number of protons.",
            "Atomic theory establishes that elemental identity and atomic number correspond to nuclear proton count, regardless of the neutron number.",
            "A chemical element's atomic number is equal to its nuclear proton charge, proving that neutron count does not determine atomic number.",
            "Isotopes of carbon have varying numbers of neutrons (6, 7, or 8) but all share the atomic number 6 because they have 6 protons.",
        ),
    ),
    (
        "Sound requires a physical medium such as air, water, or solids to propagate.",
        "science", "SUPPORTED",
        (
            "Sound consists of mechanical waves that require a material medium—gas, liquid, or solid—to transmit acoustic energy through particle oscillations.",
            "Acoustics proves that sound cannot propagate without a medium, requiring elastic interactions between neighboring molecules.",
            "Physics experiments show that sound waves travel through gases, liquids, and solids via longitudinal pressure oscillations requiring physical matter.",
            "Because sound waves are mechanical vibrations of matter, a physical medium is strictly necessary for their transmission.",
            "Acoustic pressure waves propagate through physical media by causing molecules to vibrate and collide, failing to transmit in empty space.",
        ),
    ),
    (
        "Birds evolved from a lineage of theropod dinosaurs.",
        "science", "SUPPORTED",
        (
            "Paleontological and phylogenetic evidence demonstrates that modern birds evolved from small, feathered bipedal theropod dinosaurs during the Jurassic period.",
            "Fossil discoveries such as Archaeopteryx confirm the evolutionary transition from maniraptoran theropod dinosaurs to avian species.",
            "Comparative anatomy and genomic studies show that living birds are avian dinosaurs belonging phylogenetically to the theropod clade.",
            "The presence of feathers, hollow bones, and furculae in both theropod fossils and modern birds provides conclusive proof of their evolutionary lineage.",
            "Evolutionary biologists classify birds as crown-group dinosaurs descended directly from Mesozoic theropod ancestors.",
        ),
    ),
    (
        "The Earth rotates around its axis from east to west, causing the Sun to rise in the west.",
        "science", "CONTRADICTED",
        (
            "Earth rotates on its axis from west to east (counterclockwise when viewed from above the North Pole), causing celestial bodies like the Sun to rise in the east.",
            "Astronomical observations confirm that Earth's prograde west-to-east rotation causes the diurnal apparent motion of the Sun rising in the east and setting in the west.",
            "Geophysical physics shows that Earth rotates towards the east, refuting the false claim that it rotates east to west with sunrise in the west.",
            "The rotation of planet Earth proceeds eastward, which is why morning daylight first appears over eastern longitudes as the Sun rises in the east.",
            "Planetary dynamics prove Earth rotates west to east, producing sunrise in the east across all terrestrial observation points.",
        ),
    ),
    (
        "Water molecules consist of two hydrogen atoms covalently bonded to one oxygen atom.",
        "science", "SUPPORTED",
        (
            "The chemical formula of water is H2O, consisting of two hydrogen atoms covalently bonded to a central oxygen atom with a bent molecular geometry.",
            "Molecular chemistry establishes that a water molecule comprises two single covalent bonds connecting two hydrogen atoms to one oxygen atom.",
            "Spectroscopic and structural analysis confirms that water (H2O) contains two hydrogen atoms and one oxygen atom with a bond angle of roughly 104.5 degrees.",
            "In chemical stoichiometry, each water molecule contains a 2:1 ratio of hydrogen to oxygen atoms bound by polar covalent bonds.",
            "Fundamental chemistry defines water as dihydrogen monoxide, formed by two hydrogen atoms bonded to one oxygen atom.",
        ),
    ),
    (
        "Gravity on the surface of the Moon is stronger than gravity on the surface of the Earth.",
        "science", "CONTRADICTED",
        (
            "Surface gravity on the Moon is approximately 1.62 m/s^2, which is only about one-sixth (16.5%) of Earth's surface gravity of 9.81 m/s^2.",
            "Due to its smaller mass and radius, the Moon exerts a surface gravitational pull substantially weaker than Earth's surface gravity.",
            "Apollo astronaut telemetry showed astronauts easily bounding across the lunar surface because lunar gravity is far weaker than Earth's gravity.",
            "Newtonian gravitational physics proves that Earth's greater mass produces much stronger surface gravity than that of the smaller Moon.",
            "Measurements of lunar gravitational acceleration confirm that an object weighs roughly 83% less on the Moon than on Earth.",
        ),
    ),
    (
        "Antibiotics are medications designed to treat bacterial infections rather than viral infections.",
        "science", "SUPPORTED",
        (
            "Antibiotics are pharmacological agents specifically targeting bacterial pathogens and are ineffective against viral infections such as influenza or the common cold.",
            "Medical science emphasizes that antibiotics work by inhibiting bacterial cell wall synthesis or bacterial protein production, having no effect on viruses.",
            "Clinical guidelines state that antibiotics should only be prescribed for bacterial diseases, as they lack mechanisms to kill or inhibit viruses.",
            "Pharmacology confirms that antibiotics target cellular structures unique to bacteria, rendering them completely useless against viral pathogens.",
            "Health organizations worldwide educate clinicians and patients that antibiotics treat bacterial infections and cannot cure viral ailments.",
        ),
    ),
    (
        "The ozone layer is located in the troposphere directly at sea level.",
        "science", "CONTRADICTED",
        (
            "The protective ozone layer is located primarily in the stratosphere, approximately 15 to 35 kilometers above Earth's surface, not at sea level.",
            "Atmospheric science establishes that stratospheric ozone resides high above the troposphere, absorbing harmful solar ultraviolet radiation.",
            "Ground-level ozone found at sea level is a harmful pollutant and smog component, distinct from the natural protective ozone layer in the stratosphere.",
            "Satellite sounding confirms that the ozone layer concentrated peak occurs in the middle stratosphere, refuting claims that it sits at sea level.",
            "Meteorological profiles place the ozone layer between 15 and 30 km altitude in the stratosphere, far above the sea-level troposphere.",
        ),
    ),
    (
        "Photosynthetic cyanobacteria played a primary role in Earth's Great Oxidation Event.",
        "science", "SUPPORTED",
        (
            "Geobiological evidence indicates that oxygen-producing photosynthetic cyanobacteria were responsible for the Great Oxidation Event roughly 2.4 billion years ago.",
            "The emergence of oxygenic photosynthesis in ancient cyanobacteria generated the free oxygen that transformed Earth's atmosphere during the Paleoproterozoic era.",
            "Geological records of banded iron formations demonstrate that early cyanobacterial blooms produced the atmospheric oxygen that oxygenated early Earth.",
            "Scientific consensus attributes the initial rise of atmospheric oxygen on Earth to the metabolic activity of early photosynthetic cyanobacteria.",
            "Paleontological geochemical data prove that cyanobacterial water photolysis provided the biogenic oxygen that initiated the Great Oxidation Event.",
        ),
    ),
    (
        "Mammals are cold-blooded ectothermic animals that cannot regulate their own body temperature.",
        "science", "CONTRADICTED",
        (
            "Mammals are warm-blooded endothermic vertebrates that actively regulate and maintain a constant internal body temperature through metabolic thermoregulation.",
            "Physiological zoology defines mammals as endotherms, unlike cold-blooded ectotherms such as reptiles, amphibians, and fish.",
            "Biological studies confirm that mammals maintain homeothermic internal temperatures using metabolic heat production, shivering, and sweating.",
            "Veterinary and biological consensus confirms that mammals are endothermic, disproving the notion that they are cold-blooded ectotherms.",
            "Mammalian physiology relies on high metabolic rates to maintain elevated internal body temperatures independently of ambient temperatures.",
        ),
    ),

    # =========================================================================
    # HISTORY (30 claims: 15 SUPPORTED, 15 CONTRADICTED)
    # =========================================================================
    (
        "The Declaration of Independence was adopted by the Continental Congress in the year 2010.",
        "history", "CONTRADICTED",
        (
            "On July 4, 1776, the Second Continental Congress meeting in Philadelphia formally adopted the Declaration of Independence, severing ties with Great Britain.",
            "Drafted primarily by Thomas Jefferson, the Declaration of Independence was ratified by congressional delegates in July 1776.",
            "Historical records of the Continental Congress document the adoption of the unanimous Declaration of Independence on July 4, 1776.",
            "The formal establishment of American sovereign independence began with the adoption of the Declaration of Independence in the summer of 1776.",
            "The national archives preserve the original 1776 parchment of the Declaration of Independence endorsed by delegates of the thirteen American colonies.",
        ),
    ),
    (
        "Napoleon Bonaparte won the Battle of Waterloo in June 1815.",
        "history", "CONTRADICTED",
        (
            "The Battle of Waterloo fought on June 18, 1815, resulted in the decisive defeat of Napoleon Bonaparte by Anglo-allied and Prussian coalition armies.",
            "Napoleon's French forces were routed at Waterloo by the Duke of Wellington and Field Marshal Blucher, ending Napoleon's rule as French Emperor.",
            "Defeat at Waterloo forced Napoleon's final abdication and led to his permanent exile to the remote South Atlantic island of Saint Helena.",
            "Military history documents Waterloo as Napoleon's catastrophic final defeat, terminating the Hundred Days restoration in June 1815.",
            "Allied victory at Waterloo ended the Napoleonic Wars, disproving any claim that Napoleon emerged victorious in that decisive 1815 engagement.",
        ),
    ),
    (
        "The Apollo 11 mission successfully landed humans on the Moon in July 1969.",
        "history", "SUPPORTED",
        (
            "On July 20, 1969, American astronauts Neil Armstrong and Buzz Aldrin landed the Apollo 11 Lunar Module Eagle in the Moon's Sea of Tranquility.",
            "Neil Armstrong became the first human to step onto the lunar surface on July 20, 1969, fulfilling NASA's Apollo 11 lunar landing objective.",
            "NASA telemetry and worldwide television broadcasts confirmed the successful crewed lunar landing of Apollo 11 in July 1969.",
            "Astronauts spent over two hours on the lunar surface during Apollo 11 in July 1969, gathering rock samples and deploying scientific experiments.",
            "The Apollo 11 mission realized President Kennedy's national goal of landing humans on the Moon and returning them safely before 1970.",
        ),
    ),
    (
        "The Roman Empire collapsed completely in the fifth century BC.",
        "history", "CONTRADICTED",
        (
            "The Western Roman Empire fell in 476 AD (fifth century AD), almost one thousand years after the fifth century BC, while the Eastern Empire endured until 1453 AD.",
            "In the fifth century BC, Rome was still a small burgeoning republic in central Italy, centuries before it ever became an empire.",
            "Classical history shows that the Roman Empire was founded under Augustus in 27 BC and fell in the fifth century AD, not BC.",
            "Historical chronology places the fall of Rome under the last Western emperor Romulus Augustulus in 476 AD, refuting any fifth century BC date.",
            "The Roman state expanded throughout the Mediterranean during the Hellenistic era and did not dissolve until centuries into the Common Era.",
        ),
    ),
    (
        "The Magna Carta was granted by King John of England at Runnymede in 1215.",
        "history", "SUPPORTED",
        (
            "In June 1215, King John of England affixed the Great Seal to Magna Carta at Runnymede near Windsor under pressure from rebellious English barons.",
            "Magna Carta Libertatum was issued in 1215 by King John to resolve political conflict with feudal barons, establishing principles of the rule of law.",
            "English legal history records that King John granted the charter of liberties known as Magna Carta at Runnymede meadow in June 1215.",
            "The historic peace treaty of Runnymede in 1215 produced Magna Carta, which limited royal authority and laid foundations for statutory rights.",
            "Four original surviving 1215 exemplars of Magna Carta confirm that King John granted the charter at Runnymede in the seventeenth year of his reign.",
        ),
    ),
    (
        "Julius Caesar was assassinated in the year 1945.",
        "history", "CONTRADICTED",
        (
            "Roman dictator Julius Caesar was assassinated by a conspiracy of Roman senators led by Brutus and Cassius on the Ides of March (March 15), 44 BC.",
            "Historical texts from Plutarch and Suetonius record that Caesar was stabbed 23 times in the Curia of Pompey in Rome on March 15, 44 BC.",
            "The assassination of Julius Caesar on the Ides of March in 44 BC triggered civil wars that ultimately destroyed the Roman Republic.",
            "Epigraphic and literary sources verify that Julius Caesar met his death at the hands of senatorial conspirators in 44 BC.",
            "The assassination in the Roman Senate on March 15, 44 BC ended Caesar's perpetual dictatorship and paved the way for Augustus Caesar's principate.",
        ),
    ),
    (
        "The American Civil War took place between 1914 and 1918.",
        "history", "CONTRADICTED",
        (
            "The American Civil War was fought between 1861 and 1865, whereas the period 1914 to 1918 was the duration of World War I.",
            "Historical documentation dates the American Civil War from April 1861 at Fort Sumter to April 1865 at Appomattox Court House.",
            "The conflict between the Northern Union and Southern Confederate states occurred in the 1860s, five decades before 1914.",
            "Military history clearly distinguishes the American Civil War (1861–1865) from the global conflict of the First World War (1914–1918).",
            "President Abraham Lincoln led the Union throughout the American Civil War from 1861 until his assassination in 1865, not during 1914–1918.",
        ),
    ),
    (
        "Christopher Columbus landed in the Americas in 1492 while seeking a westward route to Asia.",
        "history", "SUPPORTED",
        (
            "In August 1492, Genoese navigator Christopher Columbus sailed west across the Atlantic under the Spanish crown, making landfall in the Bahamas on October 12, 1492.",
            "Columbus's 1492 transatlantic voyage aimed to establish an oceanic trade route to the East Indies, unexpectedly reaching Caribbean islands in the Americas.",
            "Historical logs confirm that the Nina, Pinta, and Santa Maria under Columbus reached the Americas in October 1492.",
            "The 1492 expedition commanded by Christopher Columbus initiated sustained European contact and colonization across the American continents.",
            "Archival charters of the Catholic Monarchs Ferdinand and Isabella sponsored Columbus's historic 1492 voyage to find a westward sea passage.",
        ),
    ),
    (
        "Alexander the Great died in Tokyo, Japan in the year 2000.",
        "history", "CONTRADICTED",
        (
            "King Alexander the Great of Macedon died at the age of 32 in the palace of Nebuchadnezzar II in Babylon in June 323 BC.",
            "Ancient historians Arrian and Plutarch confirm that Alexander succumbed to fever and illness in Babylon in the summer of 323 BC.",
            "Following his campaigns across Persia and India, Alexander returned to Babylon, where his sudden death in 323 BC fractured his vast empire.",
            "Alexander's death in Babylon in June 323 BC initiated the Wars of the Diadochi among his Macedonian generals for imperial succession.",
            "Babylonian astronomical diaries and classical chronicles date Alexander's death precisely to the month of Daesios in 323 BC in Babylon.",
        ),
    ),
    (
        "The Berlin Wall fell and borders opened between East and West Germany in the year 1500.",
        "history", "CONTRADICTED",
        (
            "On November 9, 1989, East German officials announced that citizens were free to cross border checkpoints, leading to the peaceful fall of the Berlin Wall.",
            "Crowds of East and West Berliners gathered at the Berlin Wall on the night of November 9, 1989, chipping away at the barrier and celebrating its collapse.",
            "The opening of the Berlin Wall in November 1989 marked the collapse of Communist rule in East Germany and set in motion German reunification in 1990.",
            "Archival footage and diplomatic records document the demolition of Berlin Wall border fortifications beginning on November 9, 1989.",
            "The fall of the Berlin Wall in late 1989 symbolized the end of the Cold War and the ideological division of the European continent.",
        ),
    ),
    (
        "The United Kingdom joined the European Economic Community in 1873.",
        "history", "CONTRADICTED",
        (
            "The United Kingdom joined the European Economic Community (EEC) on January 1, 1973, exactly a century after 1873.",
            "In 1873, the European Economic Community did not exist; it was only established by the Treaty of Rome in 1957.",
            "Historical treaties confirm that Prime Minister Edward Heath signed the Treaty of Accession taking Britain into the EEC in 1973, not 1873.",
            "British entry into the European Common Market occurred in the post-WWII era in 1973, refuting any nineteenth-century accession date.",
            "Nineteenth-century Britain pursued a policy of imperial isolationism, while its modern accession to European institutions happened in 1973.",
        ),
    ),
    (
        "The French Revolution began in the year 1200.",
        "history", "CONTRADICTED",
        (
            "The French Revolution commenced in 1789, highlighted by the meeting of the Estates-General at Versailles and the storming of the Bastille prison in Paris on July 14.",
            "In 1789, popular unrest and financial crisis in France culminated in the formation of the National Assembly and the seizure of the Bastille fortress.",
            "Revolutionary events in 1789 transformed the Kingdom of France from an absolute monarchy into a constitutional regime governed by representative assemblies.",
            "French historical records identify 1789 as the watershed inception of the French Revolution, marked by the Declaration of the Rights of Man.",
            "The capture of the royal Bastille fortress by Parisian citizens on July 14, 1789 became the iconic symbol of the beginning of the French Revolution.",
        ),
    ),
    (
        "The Spanish Armada successfully conquered England and deposed Queen Elizabeth I in 1588.",
        "history", "CONTRADICTED",
        (
            "The Spanish Armada sent by King Philip II of Spain in 1588 was decisively defeated by the English navy and dispersed by severe storms, failing completely.",
            "Queen Elizabeth I remained firmly in power as the English fleet under Howard and Drake repelled the Spanish Armada in the English Channel in 1588.",
            "Far from conquering England, the Spanish Armada suffered catastrophic vessel losses along the coasts of Scotland and Ireland during its retreat.",
            "Naval history records the English victory over the Armada in 1588 as a major triumph that preserved Protestant rule under Elizabeth I.",
            "The 1588 invasion fleet failed to land the Duke of Parma's invasion army in England, disproving claims that Spain conquered England or deposed Elizabeth.",
        ),
    ),
    (
        "The ancient Library of Alexandria was located in modern-day Antarctica.",
        "history", "CONTRADICTED",
        (
            "The Great Library of Alexandria, founded in Ptolemaic Egypt in the third century BC, was one of the largest and most significant libraries of the ancient world.",
            "Scholars from across the Mediterranean gathered at the Mouseion and Library of Alexandria to study astronomy, mathematics, medicine, and literature.",
            "Ptolemy I Soter and Ptolemy II Philadelphus sponsored the Library of Alexandria, which aimed to collect the totality of Greek and Mediterranean knowledge.",
            "Historical sources confirm that Alexandria in Egypt functioned as the premier intellectual capital of the Hellenistic world through its famous library.",
            "Celebrated polymaths including Eratosthenes, Archimedes, and Aristarchus conducted pioneering intellectual work associated with the Library of Alexandria.",
        ),
    ),
    (
        "World War I ended with the signing of the Armistice on November 11, 1918.",
        "history", "SUPPORTED",
        (
            "The armistice between the Allied nations and Germany went into effect on the eleventh hour of the eleventh day of the eleventh month in 1918, halting World War I.",
            "Military operations on the Western Front ceased on November 11, 1918, following the signature of the armistice treaty in a railway carriage in Compiegne, France.",
            "The Armistice of November 11, 1918 concluded active fighting in the First World War, commemorated annually as Armistice Day.",
            "Treaty archives verify that German and Allied representatives concluded hostilities effective November 11, 1918, ending four years of trench warfare.",
            "The guns fell silent across Europe on November 11, 1918, bringing a formal close to military combat in the First World War.",
        ),
    ),
    (
        "The Taj Mahal in Agra was commissioned by the Mughal emperor Shah Jahan.",
        "history", "SUPPORTED",
        (
            "Mughal emperor Shah Jahan commissioned the white marble Taj Mahal in Agra in 1631 as a mausoleum for his beloved wife Mumtaz Mahal.",
            "Architectural records of the Mughal Empire confirm that Shah Jahan oversaw the construction of the Taj Mahal complex in Agra between 1632 and 1653.",
            "The Taj Mahal stands on the bank of the Yamuna River in Agra, constructed by thousands of artisans under imperial decrees issued by Shah Jahan.",
            "Historians recognize the Taj Mahal as the pinnacle of Mughal architecture, erected by Shah Jahan to honor the memory of Mumtaz Mahal.",
            "Inscriptions on the monument and contemporary court chronicles document Shah Jahan's direct patronage in erecting the Taj Mahal.",
        ),
    ),
    (
        "The Titanic sank on its maiden voyage in April 1912 after colliding with an iceberg.",
        "history", "SUPPORTED",
        (
            "The British ocean liner RMS Titanic struck an iceberg in the North Atlantic on the night of April 14, 1912, and sank in the early morning of April 15.",
            "On its maiden crossing from Southampton to New York City in April 1912, the White Star Line's Titanic suffered fatal hull damage from an iceberg collision.",
            "Maritime inquiry reports from 1912 established that the Titanic foundered after its starboard compartments flooded following contact with an iceberg.",
            "Over 1,500 passengers and crew perished when the Titanic sank in the cold waters of the North Atlantic Ocean in April 1912.",
            "The wreck of the Titanic on the North Atlantic seabed confirms that hull breaches caused by iceberg impact sank the vessel on its first commercial voyage.",
        ),
    ),
    (
        "George Washington was elected the first President of the United States under the Constitution.",
        "history", "SUPPORTED",
        (
            "George Washington was unanimously chosen by the Electoral College in 1789 to serve as the first President of the United States under the new federal Constitution.",
            "Following the ratification of the U.S. Constitution, General George Washington took the presidential oath of office in New York City on April 30, 1789.",
            "Constitutional archives show that George Washington served two terms as the nation's inaugural executive officer from 1789 to 1797.",
            "Historians recognize George Washington as the first President of the American republic under the constitutional framework enacted in 1788.",
            "Electoral returns certified that George Washington secured all electoral votes cast in 1789 to become the first U.S. constitutional president.",
        ),
    ),
    (
        "The Panama Canal was opened for international maritime traffic in the year 1714.",
        "history", "CONTRADICTED",
        (
            "The Panama Canal was officially opened for commercial shipping on August 15, 1914, exactly two hundred years after 1714.",
            "In the eighteenth century, engineering technology was incapable of constructing a trans-isthmian canal; the canal was completed by the United States in 1914.",
            "Historical records document the passage of the cargo ship SS Ancon as the inaugural voyage through the Panama Canal in August 1914, not 1714.",
            "The construction of the Panama Canal was a twentieth-century engineering feat completed between 1904 and 1914, disproving any eighteenth-century date.",
            "Treaty archives and diplomatic histories establish that the Panama Canal commenced operational service in 1914 following a decade of American construction.",
        ),
    ),
    (
        "The printing press using movable metal type was introduced in Europe by Johannes Gutenberg.",
        "history", "SUPPORTED",
        (
            "German goldsmith Johannes Gutenberg invented the movable-type mechanical printing press in Mainz around 1440, revolutionizing European publishing.",
            "Gutenberg's development of hand mold typecasting, oil-based ink, and a wooden screw press enabled the mass production of books in Renaissance Europe.",
            "Historical bibliography credits Johannes Gutenberg with introducing movable metal type to Europe and printing the famous Gutenberg Bible around 1455.",
            "Gutenberg's movable-type technology spread rapidly across European commercial hubs, drastically reducing book costs and facilitating the dissemination of ideas.",
            "Incunabula research confirms that Johannes Gutenberg established the first European movable metal type printing workshop in Mainz, Germany.",
        ),
    ),
    (
        "The Bolshevik Revolution in Russia took place in the year 1917.",
        "history", "SUPPORTED",
        (
            "The Bolshevik Revolution, also known as the October Revolution, led by Vladimir Lenin and Leon Trotsky, overthrew the Russian Provisional Government in 1917.",
            "In late 1917, Bolshevik armed detachments seized power in Petrograd, establishing a Soviet socialist republic and concluding the Romanov dynasty's legacy.",
            "Russian state archives date the Bolshevik socialist uprising to October 25 (Old Style) / November 7 (New Style), 1917.",
            "The 1917 Bolshevik takeover resulted in Russia's exit from World War I and the eventual establishment of the Union of Soviet Socialist Republics.",
            "Historical scholarship unanimously records 1917 as the year of the Russian Bolshevik Revolution that transformed 20th-century global politics.",
        ),
    ),
    (
        "The ancient city of Pompeii was destroyed by the eruption of Mount Vesuvius in 79 AD.",
        "history", "SUPPORTED",
        (
            "In 79 AD, the catastrophic volcanic eruption of Mount Vesuvius buried the flourishing Roman cities of Pompeii and Herculaneum under deep pumice and volcanic ash.",
            "Eyewitness accounts recorded by Pliny the Younger document the volcanic explosion of Mount Vesuvius that destroyed Pompeii in autumn 79 AD.",
            "Archaeological excavations at Pompeii reveal Roman buildings and artifacts frozen in time beneath meters of volcanic material deposited in 79 AD.",
            "Roman chronicles date the devastating eruption of Mount Vesuvius that annihilated Pompeii to the reign of Emperor Titus in 79 AD.",
            "Volcanological analysis of tephra layers confirms that pyroclastic density currents from Mount Vesuvius obliterated Pompeii in 79 AD.",
        ),
    ),
    (
        "The Suez Canal connects the Baltic Sea directly to the Black Sea.",
        "history", "CONTRADICTED",
        (
            "The Suez Canal is a sea-level artificial waterway in Egypt that connects the Mediterranean Sea to the Red Sea, not the Baltic to the Black Sea.",
            "Geographic and engineering records establish that the Suez Canal links the Mediterranean with the Gulf of Suez and the Indian Ocean across the Isthmus of Suez.",
            "The Baltic Sea and Black Sea are situated in Eastern Europe, thousands of kilometers north of the Suez Canal, which is located in northeastern Egypt.",
            "Navigation maps confirm that the Suez Canal provides a direct passage between European waters (via the Mediterranean) and Asia (via the Red Sea).",
            "Maritime records verify that the Suez Canal does not touch or connect the Baltic Sea or the Black Sea.",
        ),
    ),
    (
        "The Treaty of Versailles was signed in 1919 to formally conclude peace with Germany after World War I.",
        "history", "SUPPORTED",
        (
            "On June 28, 1919, Allied powers and Germany signed the Treaty of Versailles in the Hall of Mirrors at Versailles, formally concluding World War I.",
            "The 1919 Paris Peace Conference produced the Treaty of Versailles, which established the League of Nations and imposed terms upon post-war Germany.",
            "Diplomatic records confirm that the Treaty of Versailles was concluded on June 28, 1919, exactly five years after the assassination of Archduke Franz Ferdinand.",
            "The Covenant of the League of Nations was incorporated into the Treaty of Versailles signed by international delegations in June 1919.",
            "Historical treaties identify Versailles 1919 as the primary diplomatic agreement terminating the state of war between Germany and the Allied Powers.",
        ),
    ),
    (
        "The Inca Empire was located primarily in the Andes mountains of South America.",
        "history", "SUPPORTED",
        (
            "The Inca Empire, the largest empire in pre-Columbian America, was centered in the Andean highlands of modern-day Peru, Bolivia, Ecuador, and Chile.",
            "Inca civilization flourished along the spine of the South American Andes mountains, establishing their imperial capital at Cusco.",
            "Archaeological remnants such as Machu Picchu and an extensive 40,000-kilometer road network confirm the Inca Empire's dominion across the Andes.",
            "Ethnohistoric records and colonial chronicles place the administrative and military heart of Tawantinsuyu (the Inca Empire) in the high Andes.",
            "Geographical analysis confirms that the Inca state was an Andean mountain civilization spanning western South America prior to Spanish conquest.",
        ),
    ),
    (
        "Abraham Lincoln was the second President of the United States.",
        "history", "CONTRADICTED",
        (
            "Abraham Lincoln served as the 16th President of the United States from 1861 to 1865; John Adams was the second President (1797–1801).",
            "Presidential succession records list George Washington as the 1st president, John Adams as the 2nd president, and Abraham Lincoln as the 16th.",
            "Historical documentation shows that Abraham Lincoln was elected decades later in 1860, not as the second chief executive of the nation.",
            "Lincoln led the United States through the Civil War as its 16th president, refuting the claim that he was the second president.",
            "Official White House biographical records identify John Adams as the second president and Abraham Lincoln as the sixteenth.",
        ),
    ),
    (
        "The Battle of Gettysburg was fought during the American Civil War in July 1863.",
        "history", "SUPPORTED",
        (
            "The Battle of Gettysburg took place from July 1 to July 3, 1863, around the town of Gettysburg, Pennsylvania, during the American Civil War.",
            "General Robert E. Lee's Confederate Army of Northern Virginia clashed with General George Meade's Union Army of the Potomac at Gettysburg in July 1863.",
            "Considered the military turning point of the American Civil War, the three-day battle at Gettysburg in 1863 resulted in over 50,000 casualties.",
            "President Lincoln delivered his immortal Gettysburg Address later in November 1863 to dedicate the soldiers' national cemetery on the battlefield.",
            "Historical records and battlefield preservation confirm that Gettysburg was fought in July 1863, repelling the Confederate invasion of the North.",
        ),
    ),
    (
        "The Great Fire of London destroyed large parts of London in September 1666.",
        "history", "SUPPORTED",
        (
            "The Great Fire of London erupted on Sunday, September 2, 1666, in a bakery on Pudding Lane and swept through the city for four days.",
            "Contemporary diaries of Samuel Pepys and John Evelyn describe how the 1666 fire consumed over 13,000 houses and 87 parish churches in London.",
            "Historical municipal records show that the 1666 disaster destroyed the medieval City of London inside the old Roman city wall, including St. Paul's Cathedral.",
            "King Charles II directed firefighting efforts that finally halted the Great Fire of London on September 6, 1666.",
            "Architectural reconstruction surveys following the September 1666 blaze led to Sir Christopher Wren's rebuilding of London churches.",
        ),
    ),
    (
        "The United States purchased Alaska from the Russian Empire in 1867.",
        "history", "SUPPORTED",
        (
            "On March 30, 1867, U.S. Secretary of State William Seward negotiated the purchase of Alaska from the Russian Empire for 7.2 million dollars.",
            "The Alaska Purchase Treaty of 1867 transferred 586,412 square miles of northern territory from Czar Alexander II of Russia to the United States.",
            "Congressional records and the federal check paid to Russian minister Eduard de Stoeckl confirm the 1867 acquisition of Alaska.",
            "Initially ridiculed as Seward's Folly, the 1867 Alaska purchase added immense mineral wealth and territory to the United States.",
            "Diplomatic archives preserve the signed treaty of March 1867 establishing Russian cession of Alaska to the American government.",
        ),
    ),
    (
        "The Soviet Union dissolved peacefully on July 4, 1776.",
        "history", "CONTRADICTED",
        (
            "The Soviet Union was officially dissolved on December 26, 1991; July 4, 1776 was the date of the United States Declaration of Independence.",
            "In 1776, the Russian Empire was ruled by Empress Catherine the Great; the Soviet Union did not come into existence until December 1922.",
            "Diplomatic history records that Mikhail Gorbachev resigned as Soviet president on December 25, 1991, followed by the USSR's formal dissolution.",
            "Historical timelines place the collapse of the Soviet Union at the end of the Cold War in December 1991, over two centuries after 1776.",
            "The Declaration of the Council of the Republics of the Supreme Soviet ended the USSR in late 1991, disproving any 1776 date.",
        ),
    ),

    # =========================================================================
    # TECHNOLOGY (30 claims: 15 SUPPORTED, 15 CONTRADICTED)
    # =========================================================================
    (
        "The World Wide Web was invented in the year 1605.",
        "technology", "CONTRADICTED",
        (
            "British scientist Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN, proposing an internet-based hypertext system for scientists.",
            "Berners-Lee wrote the first web server, browser, and HTML specification at CERN between 1989 and 1990 to realize the World Wide Web.",
            "CERN archival records confirm that Tim Berners-Lee submitted Information Management: A Proposal in March 1989, creating the World Wide Web.",
            "Universal Resource Locators (URLs), HTTP, and HTML were developed by Berners-Lee at CERN to implement his 1989 web architecture.",
            "CERN released the World Wide Web source code into the public domain in 1993, accelerating global adoption of Berners-Lee's 1989 invention.",
        ),
    ),
    (
        "Bitcoin uses a proof-of-work consensus algorithm to secure its decentralized ledger.",
        "technology", "SUPPORTED",
        (
            "The Bitcoin network relies on a Proof-of-Work (PoW) consensus mechanism designed by Satoshi Nakamoto to achieve distributed Byzantine fault tolerance.",
            "Bitcoin miners perform computationally intensive SHA-256 hash calculations to solve cryptographic puzzles and append new blocks to the blockchain.",
            "Proof-of-work in Bitcoin ensures that rewriting transaction history requires controlling more than 51 percent of global hashing power, securing the ledger.",
            "The difficulty of Bitcoin's proof-of-work algorithm adjusts automatically every 2016 blocks to maintain a target block interval of ten minutes.",
            "Cryptographic mining rewards incentivize network participants to validate transactions and secure Bitcoin's blockchain through proof-of-work.",
        ),
    ),
    (
        "Relational databases store data exclusively in unstructured JSON key-value pairs.",
        "technology", "CONTRADICTED",
        (
            "Relational database management systems (RDBMS) organize data strictly into structured tables composed of predefined rows and columns governed by schemas.",
            "Codd's relational model uses mathematical relations with primary and foreign keys, whereas unstructured JSON key-value storage defines NoSQL document stores.",
            "Structured Query Language (SQL) operates on relational tabular relations with strict schema constraints, not unstructured JSON key-value pairs.",
            "While modern relational databases offer JSON extensions, their foundational architecture is based on relational table schemas, not unstructured key-value stores.",
            "Document databases like MongoDB store JSON documents, whereas traditional relational systems like PostgreSQL and MySQL are built around tabular relations.",
        ),
    ),
    (
        "Machine learning models commonly optimize loss functions using gradient descent algorithms.",
        "technology", "SUPPORTED",
        (
            "Gradient descent is the primary optimization algorithm used to train machine learning models by iteratively updating parameters in the direction of steepest descent.",
            "Backpropagation calculates partial derivatives of the loss function with respect to neural network weights, guiding gradient descent parameter updates.",
            "Variants such as Stochastic Gradient Descent (SGD) and Adam adaptively adjust learning rates to minimize empirical loss during deep model training.",
            "Mathematical loss functions quantify the discrepancy between model predictions and target labels, providing a scalar objective minimized via gradient descent.",
            "Optimization algorithms in machine learning rely on calculus-based gradient vectors to steer model weights toward local minima of the loss surface.",
        ),
    ),
    (
        "Ethernet was conceived as an acoustic steam-powered system in 1800.",
        "technology", "CONTRADICTED",
        (
            "Ethernet computer networking technology was conceived by Robert Metcalfe in 1973 at Xerox Palo Alto Research Center (PARC) to interconnect workstations.",
            "Metcalfe's memo at Xerox PARC outlined the principles of carrier sense multiple access with collision detection (CSMA/CD) that defined original Ethernet.",
            "The first experimental Ethernet network connected Xerox Alto personal computers to laser printers at PARC at an initial transmission rate of 2.94 Mbps.",
            "Xerox collaborated with DEC and Intel to standardize the DIX Ethernet specification in 1980, commercializing Metcalfe's Xerox PARC invention.",
            "Robert Metcalfe later founded 3Com to commercialize Ethernet adapters, establishing the dominant local area networking standard born at Xerox PARC.",
        ),
    ),
    (
        "The Linux kernel was created in the fourteenth century.",
        "technology", "CONTRADICTED",
        (
            "Finnish computer science student Linus Torvalds created the Linux operating system kernel and posted its initial version 0.01 release in September 1991.",
            "Linus Torvalds announced his free operating system project on the Minix newsgroup in August 1991, establishing the foundation for the open-source Linux kernel.",
            "Historical software repositories document Torvalds' release of the Linux kernel under the GNU General Public License in 1991.",
            "The Linux monolithic kernel was developed by Linus Torvalds in 1991 for Intel 80386 personal computers, later expanding across global computing platforms.",
            "Torvalds remains the principal maintainer of the Linux kernel, whose development originated with his personal open-source code release in 1991.",
        ),
    ),
    (
        "Solid state drives (SSDs) use mechanical spinning magnetic platters to store binary data.",
        "technology", "CONTRADICTED",
        (
            "Solid-state drives (SSDs) store data electronically using non-volatile NAND flash memory chips and have no moving mechanical parts or spinning platters.",
            "Hard disk drives (HDDs) use mechanical spinning magnetic platters, whereas SSDs rely entirely on semiconductor flash memory.",
            "Hardware engineering textbooks distinguish SSDs by their solid-state silicon architecture, contrasting them with mechanical magnetic disk drives.",
            "SSDs read and write data through integrated circuit memory cells, disproving the false claim that they contain mechanical spinning platters.",
            "Because SSDs have zero rotating platters or read/write actuator arms, they deliver much faster random access times than magnetic hard drives.",
        ),
    ),
    (
        "The HTTP protocol operates as an application layer protocol in the Internet protocol suite.",
        "technology", "SUPPORTED",
        (
            "Hypertext Transfer Protocol (HTTP) is an application layer protocol in the Internet protocol suite used for transmitting hypermedia documents on the Web.",
            "IETF standards specify HTTP as an application-level request-response protocol running over reliable transport protocols like TCP or QUIC.",
            "In the OSI reference model, HTTP functions at Layer 7 (the Application Layer), coordinating client-server communication between web browsers and servers.",
            "Network specifications confirm that HTTP operates at the application layer, relying on underlying transport layer protocols to route packets.",
            "Application protocols including HTTP, FTP, and SMTP provide high-level services directly to user software across IP networks.",
        ),
    ),
    (
        "Public-key cryptography uses the exact same single secret key for both encryption and decryption.",
        "technology", "CONTRADICTED",
        (
            "Public-key cryptography (asymmetric cryptography) uses a mathematically linked key pair: a public key for encryption and a distinct private key for decryption.",
            "Symmetric cryptography uses a single shared secret key, whereas asymmetric public-key systems strictly require two distinct keys.",
            "Cryptographic protocols like RSA utilize one public key to encrypt data and a separate private key to decrypt it, refuting single-key claims.",
            "The defining property of public-key cryptography is the asymmetry between the encryption key and the decryption key.",
            "Diffie and Hellman introduced asymmetric cryptography specifically to eliminate the need for sharing a single secret key between communicating parties.",
        ),
    ),
    (
        "The C programming language was developed by Isaac Newton in 1687.",
        "technology", "CONTRADICTED",
        (
            "Dennis Ritchie created the C programming language at AT&T Bell Laboratories between 1969 and 1973 to rewrite the Unix operating system.",
            "Computer science history documents Dennis Ritchie's design of C at Bell Labs as a successor to Ken Thompson's B programming language.",
            "Ritchie's development of C at Bell Labs in the early 1970s provided structured control flow, pointer arithmetic, and hardware portability for Unix.",
            "The classic reference The C Programming Language was co-authored by Dennis Ritchie and Brian Kernighan based on Ritchie's Bell Labs work.",
            "Bell Laboratories archives credit Dennis Ritchie with developing C, which became one of the most widely used programming languages in history.",
        ),
    ),
    (
        "Transformer neural networks utilize self-attention mechanisms to process sequential data in parallel.",
        "technology", "SUPPORTED",
        (
            "The Transformer architecture introduced by Vaswani et al. in 2017 replaces recurrence with multi-head self-attention, processing all sequence tokens in parallel.",
            "Self-attention in Transformers computes relational weights between all token pairs simultaneously, eliminating sequential recurrent hidden state constraints.",
            "Natural language processing models based on Transformers leverage parallelized self-attention to capture long-range contextual dependencies efficiently.",
            "Unlike recurrent neural networks that process tokens sequentially step-by-step, Transformers compute attention matrices across entire input sequences at once.",
            "The seminal 2017 paper Attention Is All You Need proved that self-attention mechanisms alone enable superior sequence transduction with massive training parallelism.",
        ),
    ),
    (
        "HTML is a compiled programming language executed directly by computer microprocessors.",
        "technology", "CONTRADICTED",
        (
            "HyperText Markup Language (HTML) is a declarative markup language interpreted by web browsers, not a compiled programming language executed by CPUs.",
            "HTML defines document structure and semantics using tags and elements, lacking computational logic, control flow, or native machine code compilation.",
            "Computer science taxonomy classifies HTML as markup, disproving the claim that it is a compiled microprocessor-executable programming language.",
            "Web browsers parse and render HTML into the Document Object Model (DOM); microprocessors never execute raw HTML machine instructions.",
            "Unlike languages such as C, Rust, or Fortran, HTML cannot perform calculations or compile into binary machine instructions for direct CPU execution.",
        ),
    ),
    (
        "Git is a version control system created in the year 1800.",
        "technology", "CONTRADICTED",
        (
            "Linus Torvalds designed and implemented the Git distributed version control system in 2005 to manage the Linux kernel development codebase.",
            "Git was released by Linus Torvalds in April 2005 as a decentralized, cryptographic tool prioritizing speed, data integrity, and branching flexibility.",
            "Software engineering history records Torvalds' creation of Git in 2005 after the Linux project lost access to the proprietary BitKeeper system.",
            "Git's distributed repository architecture allows every developer to hold a complete local copy of project history, designed by Torvalds in 2005.",
            "Open-source commit logs preserve Linus Torvalds' initial code commits establishing the Git revision tracking system in spring 2005.",
        ),
    ),
    (
        "Bluetooth is a wired networking standard that requires physical copper cables to connect devices.",
        "technology", "CONTRADICTED",
        (
            "Bluetooth is a short-range wireless communication technology that uses UHF radio waves, eliminating the need for physical copper cables.",
            "The Bluetooth standard operates in the 2.4 GHz ISM wireless radio band, designed specifically to replace cable connections between electronic devices.",
            "Telecommunications engineering defines Bluetooth as a personal area wireless radio frequency protocol, not a wired cable standard.",
            "Devices such as wireless headphones and keyboards connect via Bluetooth radio transceivers, refuting claims of physical copper wire dependency.",
            "The Bluetooth Special Interest Group specifications confirm that Bluetooth communication is entirely wireless over short-range radio frequencies.",
        ),
    ),
    (
        "The transistor was invented during the ancient Roman Empire.",
        "technology", "CONTRADICTED",
        (
            "Physicists John Bardeen, Walter Brattain, and William Shockley invented the point-contact transistor at AT&T's Bell Laboratories in December 1947.",
            "Bell Labs engineers demonstrated the first working solid-state transistor amplifier on December 23, 1947, revolutionizing electronic circuits.",
            "The invention of the point-contact germanium transistor at Bell Laboratories in 1947 inaugurated the modern semiconductor electronics era.",
            "Historical patent records document Bell Telephone Laboratories' December 1947 breakthrough replacing vacuum tubes with solid-state transistors.",
            "Bardeen, Brattain, and Shockley shared the 1956 Nobel Prize in Physics for their 1947 invention of the semiconductor transistor at Bell Labs.",
        ),
    ),
    (
        "RAM stands for Random Access Memory in computer architecture.",
        "technology", "SUPPORTED",
        (
            "In computing, the acronym RAM stands for Random Access Memory, referring to volatile system memory where data can be accessed in any random order.",
            "Computer engineering textbooks define RAM as Random Access Memory, providing rapid read and write access to running processor instructions.",
            "Hardware architecture establishes that RAM (Random Access Memory) serves as the primary working memory for central processing units.",
            "The term Random Access Memory (RAM) distinguishes this storage from sequential access media like magnetic tapes where access depends on mechanical position.",
            "Technical glossaries confirm that RAM is the universal abbreviation for Random Access Memory across all computing platforms.",
        ),
    ),
    (
        "Python is an interpreted, high-level programming language created by Guido van Rossum.",
        "technology", "SUPPORTED",
        (
            "Dutch programmer Guido van Rossum conceived and created the Python programming language, releasing the initial version in February 1991.",
            "Python is widely classified as an interpreted, dynamically typed, high-level general-purpose language designed by Guido van Rossum.",
            "Historical documentation shows Van Rossum developed Python at CWI in the Netherlands as a successor to the ABC programming language.",
            "Python's emphasis on code readability and clean syntax reflects Guido van Rossum's foundational language design principles.",
            "Software history credits Guido van Rossum as Python's creator and its long-serving Benevolent Dictator for Life until his retirement in 2018.",
        ),
    ),
    (
        "Modern smartphones communicate over cellular networks using analog acoustic telegraph signals.",
        "technology", "CONTRADICTED",
        (
            "Modern smartphones transmit digital voice and data over radio frequency electromagnetic spectrum, not through analog acoustic telegraph signals.",
            "Cellular wireless networks (4G LTE, 5G NR) utilize digital modulation schemes like QAM and OFDM across gigahertz radio bands.",
            "Telecommunications engineering confirms that smartphones encode information as digital binary packets sent over RF antennas, not acoustic telegraphy.",
            "Acoustic telegraph signals are obsolete nineteenth-century mechanical sound methods, entirely unrelated to digital cellular mobile telephony.",
            "Smartphone transceivers operate on microwave radio bands (600 MHz to 39 GHz), proving they do not use acoustic telegraphy.",
        ),
    ),
    (
        "The Domain Name System (DNS) translates human-readable domain names into numerical IP addresses.",
        "technology", "SUPPORTED",
        (
            "The Domain Name System (DNS) is a hierarchical naming system that translates human-friendly domain names like example.com into machine-readable IP addresses.",
            "Network specifications define DNS as the core internet directory service mapping alphanumeric hostnames to binary IPv4 and IPv6 numerical addresses.",
            "RFC 1034 and RFC 1035 specify how DNS resolver servers query distributed databases to resolve domain names to corresponding numerical IP routes.",
            "Without DNS, internet users would be required to memorize numerical IP strings rather than standard web URLs to access servers.",
            "Internet engineering consensus identifies DNS as the foundational translation service converting alphanumeric host identifiers into routing IP addresses.",
        ),
    ),
    (
        "Quantum computers process information using classical bits that can only be 0 or 1 simultaneously.",
        "technology", "CONTRADICTED",
        (
            "Quantum computers process information using quantum bits (qubits), which can exist in superpositions of 0 and 1 states simultaneously, unlike classical bits.",
            "Classical digital computers rely on binary bits constrained to either 0 or 1, whereas quantum computers exploit quantum superposition and entanglement.",
            "Quantum information theory defines qubits as continuous linear combinations of basis states, refuting the claim that they behave like classical binary bits.",
            "Experimental quantum computing utilizes physical qubits that undergo quantum interference across multi-state Hilbert spaces.",
            "The foundational advantage of quantum computing arises from qubits existing in superposition, which directly contradicts classical binary bit constraints.",
        ),
    ),
    (
        "An operating system manages computer hardware resources and provides common services for software applications.",
        "technology", "SUPPORTED",
        (
            "An operating system (OS) is system software that manages computer hardware components, processor scheduling, memory allocation, and file systems.",
            "Computer science curricula define operating systems as the software intermediary managing physical device hardware on behalf of user applications.",
            "Core operating system functions include CPU multitasking, virtual memory management, device driver abstraction, and security privilege separation.",
            "Software engineering texts establish that operating systems like Windows, Linux, and macOS arbitrate hardware access for application programs.",
            "The fundamental role of an operating system kernel is multiplexing system resources, ensuring stable, concurrent hardware utilization across processes.",
        ),
    ),
    (
        "IPv4 addresses are 128-bit hexadecimal numbers formatted with colons.",
        "technology", "CONTRADICTED",
        (
            "IPv4 addresses are 32-bit numerical addresses formatted in four dot-decimal octets (e.g. 192.168.1.1); 128-bit colon-hexadecimal addresses define IPv6.",
            "Internet Protocol Version 4 uses exactly 32 bits of address space, providing approximately 4.3 billion unique network addresses, not 128 bits.",
            "Network RFC 791 defines IPv4 addressing strictly as a 32-bit integer, while 128-bit hexadecimal formatting was introduced in IPv6 (RFC 2460).",
            "Computer networking standards distinguish 32-bit IPv4 dot-decimal notation from the 128-bit colon-separated hexadecimal notation of IPv6.",
            "The 128-bit hexadecimal structure is exclusive to IPv6, proving that IPv4 addresses are not 128-bit numbers.",
        ),
    ),
    (
        "Compilers translate high-level programming language source code into lower-level machine or intermediate code.",
        "technology", "SUPPORTED",
        (
            "A compiler is a computer program that translates source code written in a high-level programming language into target machine code, bytecode, or assembly language.",
            "Computer science theory defines compilers as translation pipelines performing lexical analysis, parsing, semantic analysis, and target code generation.",
            "Compilers like GCC, Clang, and rustc convert human-readable source syntax into binary machine code instructions executable by specific hardware CPUs.",
            "The essential purpose of a software compiler is converting high-level abstractions into executable binary representations preserving source program semantics.",
            "Textbooks on compiler design document the formal algorithms transforming high-level languages into optimized low-level machine instructions.",
        ),
    ),
    (
        "The RSA cryptosystem relies on the practical difficulty of factoring large composite prime products.",
        "technology", "SUPPORTED",
        (
            "The RSA public-key cryptographic algorithm is based on the computational intractability of the prime factorization problem for large semiprime numbers.",
            "Invented by Rivest, Shamir, and Adleman in 1977, RSA security relies on the fact that multiplying two large primes is easy, but factoring their product is hard.",
            "Mathematical cryptography establishes that decrypting RSA ciphertext without the private key requires solving integer factorization of composite modulus n.",
            "Current number-theoretic algorithms cannot factor sufficiently large 2048-bit composite numbers in polynomial time, providing empirical security for RSA.",
            "Cryptographic standards affirm that RSA encryption and digital signature security depends directly on the difficulty of integer prime factorization.",
        ),
    ),
    (
        "TCP guarantees reliable, ordered packet delivery over computer networks through acknowledgments and retransmissions.",
        "technology", "SUPPORTED",
        (
            "The Transmission Control Protocol (TCP) provides connection-oriented, reliable, and ordered delivery of data streams between network applications.",
            "RFC 793 defines TCP's mechanisms, including sequence numbers, acknowledgments, timeout retransmissions, and flow control to guarantee delivery.",
            "Unlike UDP which is connectionless and best-effort, TCP ensures that lost packets are detected and retransmitted, maintaining intact byte order.",
            "Network engineering principles classify TCP as a reliable transport protocol that recovers from packet loss and corruption across IP networks.",
            "TCP state machines maintain end-to-end reliability by requiring receiver acknowledgments and resending dropped segments across the connection.",
        ),
    ),
    (
        "All computer programs written in JavaScript require pre-compilation into standalone x86 assembly binaries.",
        "technology", "CONTRADICTED",
        (
            "JavaScript is a dynamically typed scripting language typically interpreted or just-in-time (JIT) compiled inside web browsers and runtime engines like Node.js.",
            "JavaScript code does not require pre-compilation into standalone x86 assembly binaries before execution; engines execute it directly from text source.",
            "Web browsers parse and execute JavaScript scripts on the fly using internal bytecode interpreters and JIT optimizers, not ahead-of-time x86 compilers.",
            "Software engineering practices distribute JavaScript as plaintext source files rather than pre-compiled standalone binary executables.",
            "The ECMAScript specification defines JavaScript as an interpreted scripting language, contradicting the claim that it requires pre-compiled x86 binaries.",
        ),
    ),
    (
        "Microprocessors use arithmetic logic units (ALUs) to perform mathematical and bitwise operations.",
        "technology", "SUPPORTED",
        (
            "An arithmetic logic unit (ALU) is a fundamental digital circuit within a computer processor that carries out arithmetic calculations and logical bitwise operations.",
            "Microprocessor architecture texts identify the ALU as the execution core responsible for addition, subtraction, AND, OR, and bit shift operations.",
            "CPU control units feed operand registers into the ALU, which computes results based on the decoded instruction opcode.",
            "Digital electronics confirms that ALUs perform binary addition, logic evaluations, and comparison operations inside central processing units.",
            "Every general-purpose computer processor contains one or more ALUs dedicated to executing integer arithmetic and boolean logical functions.",
        ),
    ),
    (
        "Fiber optic cables transmit internet data using pulses of light through flexible glass or plastic strands.",
        "technology", "SUPPORTED",
        (
            "Fiber optic communication transmits digital information as pulses of infrared or visible light through thin strands of silica glass or optical plastic.",
            "Telecommunications networks utilize optical fiber cables exploiting total internal reflection to carry high-bandwidth data signals over long distances.",
            "Optical engineering confirms that fiber cables guide modulated light waves produced by semiconductor lasers or LEDs with minimal signal attenuation.",
            "High-speed internet backbones rely extensively on submarine and terrestrial fiber optic cables transmitting optical photonic data streams.",
            "Physical measurements show optical fiber cables achieve immense bandwidth by transmitting modulated light frequencies through dielectric glass cores.",
        ),
    ),
    (
        "SQL injection is an attack technique that exploits vulnerabilities in database query input sanitization.",
        "technology", "SUPPORTED",
        (
            "SQL injection is a web security vulnerability allowing attackers to interfere with backend queries by inserting malicious SQL syntax into user input fields.",
            "Cybersecurity standards identify SQL injection as an input validation flaw where untrusted data is directly concatenated into dynamic database statements.",
            "OWASP classifies SQL injection among critical security risks, mitigated through parameterized queries, prepared statements, and input escaping.",
            "Vulnerability analysis shows SQL injection allows unauthorized parties to read, modify, or delete sensitive database table records.",
            "Secure coding guidelines require parameterized queries specifically to prevent attackers from executing arbitrary SQL commands via unescaped inputs.",
        ),
    ),
    (
        "Virtual memory allows a computer to address more memory than is physically installed in hardware RAM.",
        "technology", "SUPPORTED",
        (
            "Virtual memory is an operating system memory management technique that maps virtual address spaces to physical storage, extending available memory beyond RAM.",
            "By paging inactive memory chunks to secondary disk storage, virtual memory enables computers to execute programs larger than physical RAM capacity.",
            "Computer architecture textbooks explain how memory management units (MMUs) translate virtual pages into physical frames, abstracting physical RAM limits.",
            "Operating systems utilize virtual memory paging mechanisms to provide each process with a large, contiguous, and isolated address space.",
            "The implementation of virtual memory prevents out-of-memory crashes by temporarily swapping memory blocks between physical RAM and disk swap partitions.",
        ),
    ),

    # =========================================================================
    # NEWS & CURRENT AFFAIRS (30 claims: 15 SUPPORTED, 15 CONTRADICTED)
    # =========================================================================
    (
        "The European Union was formally established in the fifth century BC.",
        "news", "CONTRADICTED",
        (
            "The Treaty on European Union, signed in Maastricht in February 1992, entered into force on November 1, 1993, formally creating the European Union.",
            "Maastricht established the three-pillar institutional structure of the European Union and paved the way for European citizenship and the single currency in 1993.",
            "Ratification of the Maastricht Treaty by member nations transformed the European Community into the European Union on November 1, 1993.",
            "European diplomatic history marks November 1993 as the formal inception of the European Union following the enactment of the Maastricht accord.",
            "Legal treaties confirm the European Union came into legal existence in 1993 upon the completion of all state ratifications of the Maastricht treaty.",
        ),
    ),
    (
        "The 2020 Summer Olympic Games in Tokyo were held without delay in July 2020.",
        "news", "CONTRADICTED",
        (
            "The 2020 Tokyo Olympic Games were officially postponed in March 2020 due to the global COVID-19 pandemic, the first postponement in modern Olympic history.",
            "The International Olympic Committee rescheduled the Tokyo Games to July and August 2021, holding the competitions a full year late without spectators.",
            "Global public health lockdowns forced international athletic organizers to postpone the Tokyo 2020 Olympics until summer 2021.",
            "Olympic competition took place from July 23 to August 8, 2021, proving the Games were not held without delay in 2020.",
            "Official IOC records document that the Tokyo 2020 Games were delayed by twelve months due to international coronavirus restrictions.",
        ),
    ),
    (
        "The Paris Agreement on climate change was negotiated in the year 1800.",
        "news", "CONTRADICTED",
        (
            "At the 21st Conference of the Parties (COP21) in Paris in December 2015, 196 nations adopted the historic Paris Agreement on climate change mitigation.",
            "The Paris Climate Accord was agreed by international consensus on December 12, 2015, aiming to keep global temperature rise well below 2 degrees Celsius.",
            "United Nations Framework Convention on Climate Change delegates formally approved the landmark Paris Agreement at the close of COP21 in 2015.",
            "Diplomats from around the globe concluded negotiations on the international Paris climate treaty in Le Bourget, France, in December 2015.",
            "The historic consensus treaty establishing nationally determined emissions reduction pledges was adopted in Paris in December 2015.",
        ),
    ),
    (
        "NASA retired its Space Shuttle fleet from operational service in the year 1500.",
        "news", "CONTRADICTED",
        (
            "NASA officially ended its 30-year Space Shuttle program on July 21, 2011, when Space Shuttle Atlantis completed mission STS-135.",
            "The Space Shuttle fleet, including Discovery, Endeavour, and Atlantis, was decommissioned in 2011 and placed in museums across the United States.",
            "Following the final landing of Atlantis in July 2011, NASA retired the Space Shuttle fleet and transitioned to commercial crew partnerships.",
            "Flight STS-135 in July 2011 concluded 135 orbital shuttle missions, marking the formal retirement of NASA's winged orbiter fleet.",
            "Aerospace archives document the retirement of the U.S. Space Shuttle program in 2011 after three decades of low-Earth orbit operations.",
        ),
    ),
    (
        "The World Health Organization is headquartered in Paris, France.",
        "news", "CONTRADICTED",
        (
            "The World Health Organization (WHO), the specialized agency of the United Nations responsible for international public health, is headquartered in Geneva, Switzerland.",
            "Established on April 7, 1948, the WHO directs international health governance from its permanent headquarters complex located in Geneva, Switzerland, not Paris.",
            "Diplomatic treaties designate Geneva, Switzerland as the official seat of the World Health Organization headquarters.",
            "While UNESCO is based in Paris, the World Health Organization has maintained its global operational headquarters in Geneva, Switzerland since inception.",
            "Avenue Appia in Geneva, Switzerland hosts the global headquarters of the World Health Organization, refuting claims that it is in Paris.",
        ),
    ),
    (
        "The United Kingdom formally left the European Union in the year 1776.",
        "news", "CONTRADICTED",
        (
            "The United Kingdom officially withdrew from the European Union on January 31, 2020, following the ratification of the EU-UK Withdrawal Agreement.",
            "Brexit took formal legal effect at 23:00 GMT on January 31, 2020, ending 47 years of British membership in the European bloc.",
            "British and European parliamentary ratifications concluded the Article 50 divorce process on January 31, 2020, enacting the UK's formal departure.",
            "Government records confirm the UK's departure from the European Union occurred on January 31, 2020, entering an eleven-month transition period.",
            "International media documented the UK's ceremonial exit from the European Union at midnight Central European Time on January 31, 2020.",
        ),
    ),
    (
        "The Panama Canal expansion project opening third shipping lanes was completed in 1492.",
        "news", "CONTRADICTED",
        (
            "The expanded Panama Canal, featuring larger third-lane locks capable of handling Neopanamax vessels, was inaugurated on June 26, 2016.",
            "Panama Canal Authority officials celebrated the commercial opening of the expansion locks in June 2016 after nine years of engineering construction.",
            "Commercial shipping through the newly widened third set of locks in the Panama Canal commenced with the transit of the container ship Cosco Shipping Panama in 2016.",
            "Maritime industry reports confirmed the doubling of the Panama Canal's cargo capacity upon the formal completion of the third-lane expansion project in 2016.",
            "The Panama Canal expansion project concluded in 2016, accommodating larger interoceanic bulk carriers and container ships.",
        ),
    ),
    (
        "The United Nations was founded immediately following the conclusion of the American Civil War in 1865.",
        "news", "CONTRADICTED",
        (
            "The United Nations was founded on October 24, 1945, in the aftermath of World War II, eighty years after the 1865 American Civil War.",
            "International treaty archives confirm the UN Charter was drafted at the San Francisco Conference in 1945, not in 1865.",
            "The United Nations was established in October 1945 by 51 founding member states to replace the defunct League of Nations after World War II.",
            "Historical chronology places the creation of the United Nations in the mid-twentieth century (1945), disproving any 1865 founding date.",
            "The United Nations Charter came into force on October 24, 1945, reflecting the post-WWII geopolitical order rather than the 1865 Civil War era.",
        ),
    ),
    (
        "The euro currency was officially introduced as an accounting currency in the year 1200.",
        "news", "CONTRADICTED",
        (
            "The euro was launched on world financial markets as an electronic accounting currency on January 1, 1999, and physical euro coins and banknotes entered circulation on January 1, 2002.",
            "European central banking records show the euro replaced national currencies in non-cash transactions in 1999 before physical cash roll-out in 2002.",
            "Eleven European Union member states adopted the euro for electronic payments in January 1999, followed by cash distribution in January 2002.",
            "Economic history documents the two-phase introduction of the euro: digital banking denomination in 1999 and physical legal tender in 2002.",
            "The European Central Bank introduced physical euro notes and coins on New Year's Day 2002, completing the transition initiated in 1999.",
        ),
    ),
    (
        "The International Space Station was built and inhabited in the year 1650.",
        "news", "CONTRADICTED",
        (
            "The International Space Station (ISS) has maintained continuous human presence since Expedition 1 docked with the orbital outpost on November 2, 2000.",
            "Astronauts and cosmonauts have permanently staffed the International Space Station without interruption since November 2000.",
            "NASA and Roscosmos mission logs confirm uninterrupted human habitation aboard the ISS spanning over two decades since late 2000.",
            "Expedition 1 members William Shepherd, Yuri Gidzenko, and Sergei Krikalev inaugurated permanent continuous habitation of the ISS in November 2000.",
            "Space agency records celebrate over twenty consecutive years of continuous human occupation aboard the International Space Station since November 2000.",
        ),
    ),
    (
        "The North Atlantic Treaty Organization (NATO) was established by the Treaty of Washington in 1949.",
        "news", "SUPPORTED",
        (
            "The North Atlantic Treaty Organization (NATO) was established on April 4, 1949, when twelve founding member nations signed the North Atlantic Treaty in Washington, D.C.",
            "NATO's collective defense alliance was founded in April 1949 under the leadership of the United States and Western European allies.",
            "Treaty archives confirm that the North Atlantic Treaty was signed in Washington in 1949 to provide collective security against Soviet expansion.",
            "Article 5 collective security commitments were inaugurated with the signing of the Washington Treaty creating NATO in April 1949.",
            "Twelve sovereign nations ratified the original North Atlantic alliance charter in Washington in 1949, founding the NATO defense pact.",
        ),
    ),
    (
        "The global human population reached 8 billion individuals according to the United Nations in late 2022.",
        "news", "SUPPORTED",
        (
            "The United Nations Department of Economic and Social Affairs officially recognized November 15, 2022, as the Day of 8 Billion, when world population surpassed 8 billion.",
            "Demographic reports released by the United Nations confirmed that global human population crossed the historic milestone of 8 billion in late 2022.",
            "UN demographic tracking celebrated the birth of the eight-billionth living human on November 15, 2022, reflecting global longevity gains.",
            "Census and demographic modeling by international bodies documented the attainment of 8 billion global inhabitants in the final months of 2022.",
            "United Nations population statistics published in the 2022 World Population Prospects confirmed the 8 billion milestone occurred in November 2022.",
        ),
    ),
    (
        "The Nobel Prizes were established according to the 1895 will of Swedish inventor Alfred Nobel.",
        "news", "SUPPORTED",
        (
            "Swedish chemist and dynamite inventor Alfred Nobel signed his last will and testament in Paris on November 27, 1895, dedicating his fortune to establish the Nobel Prizes.",
            "The Nobel Foundation was created to execute Alfred Nobel's 1895 bequest endowing annual prizes in physics, chemistry, medicine, literature, and peace.",
            "Historical probate documents confirm that the Nobel Prizes originated from Alfred Nobel's 1895 testamentary bequest.",
            "The first Nobel Prizes were awarded in 1901 pursuant to the explicit instructions left in Alfred Nobel's 1895 will.",
            "Swedish legal archives preserve Alfred Nobel's 1895 will creating the international awards recognizing achievements conferring the greatest benefit to humankind.",
        ),
    ),
    (
        "The Chernobyl nuclear power plant disaster occurred in Ukraine in April 1986.",
        "news", "SUPPORTED",
        (
            "On April 26, 1986, Reactor No. 4 at the Chernobyl nuclear power plant near Pripyat in the Ukrainian Soviet Republic exploded during a safety test.",
            "The Chernobyl disaster of April 1986 released massive quantities of radioactive isotopes across Europe, becoming the worst nuclear disaster in history.",
            "Soviet and international nuclear safety reports confirm that catastrophic power surges caused the destruction of the Chernobyl reactor in April 1986.",
            "Environmental fallout from the Chernobyl explosion in late April 1986 forced the permanent evacuation of over 100,000 residents around Pripyat, Ukraine.",
            "International atomic energy archives document the Chernobyl reactor core meltdown and explosion occurring in Soviet Ukraine in April 1986.",
        ),
    ),
    (
        "The International Court of Justice has its permanent judicial seat located in Tokyo, Japan.",
        "news", "CONTRADICTED",
        (
            "The International Court of Justice (ICJ), the principal judicial organ of the United Nations, has its seat at the Peace Palace in The Hague, Netherlands.",
            "Under Article 22 of the ICJ Statute, the permanent seat of the International Court of Justice is established in The Hague, Netherlands, not Tokyo.",
            "Diplomatic treaties place the World Court in The Hague, Netherlands, where it adjudicates legal disputes between sovereign nations.",
            "While international tribunals have convened elsewhere, the permanent International Court of Justice resides exclusively in The Hague, Netherlands.",
            "United Nations charters confirm that The Hague in the Netherlands has served as the permanent judicial seat of the ICJ since its 1945 establishment.",
        ),
    ),
    (
        "The Human Genome Project announced the completion of an initial draft sequence of the human genome in 2000.",
        "news", "SUPPORTED",
        (
            "In June 2000, leaders of the public Human Genome Project and Celera Genomics jointly announced the completion of the first working draft sequence of the human genome.",
            "A White House ceremony on June 26, 2000, attended by President Clinton and Prime Minister Blair, celebrated the draft sequencing of the human genome.",
            "Scientific publications in Nature and Science in early 2001 followed the historic June 2000 announcement of the draft human genome sequence.",
            "Genomic milestone archives record that international consortia completed the initial working draft map of the three billion human DNA base pairs in 2000.",
            "The milestone completion of the draft human genome sequence was formally declared to the worldwide scientific community in the summer of 2000.",
        ),
    ),
    (
        "The Antarctic Treaty of 1959 preserves Antarctica exclusively for military colonization and commercial mining.",
        "news", "CONTRADICTED",
        (
            "The Antarctic Treaty signed in Washington in 1959 explicitly demilitarizes Antarctica, prohibiting military activity and reserving the continent for peaceful scientific research.",
            "Article 1 of the Antarctic Treaty mandates that Antarctica shall be used for peaceful purposes only, strictly forbidding military fortifications and maneuvers.",
            "The 1991 Environmental Protocol to the Antarctic Treaty explicitly bans all commercial mineral resource activities and mining in Antarctica.",
            "Diplomatic consensus preserves Antarctica as a natural scientific reserve, directly contradicting claims of military colonization or commercial mining.",
            "Treaty archives verify that Antarctica is legally protected against territorial militarization and commercial exploitation under the 1959 treaty framework.",
        ),
    ),
    (
        "The Eurotunnel connecting Great Britain and France beneath the English Channel officially opened in 1994.",
        "news", "SUPPORTED",
        (
            "The Channel Tunnel, or Eurotunnel, a 50-kilometer undersea rail tunnel connecting Folkestone in England with Coquelles in France, was inaugurated in May 1994.",
            "Queen Elizabeth II and French President Francois Mitterrand formally opened the Channel Tunnel rail connection on May 6, 1994.",
            "Commercial passenger and freight train services through the sub-Channel Eurotunnel commenced operations in 1994.",
            "Engineering records document the historic completion and opening of the fixed rail link beneath the English Channel in May 1994.",
            "The Eurotunnel began direct passenger train transits between London and Paris via the undersea tunnel system in late 1994.",
        ),
    ),
    (
        "The International Monetary Fund (IMF) was conceived at the Bretton Woods conference in July 1944.",
        "news", "SUPPORTED",
        (
            "Delegates from 44 nations negotiated the creation of the International Monetary Fund (IMF) at the UN Monetary and Financial Conference in Bretton Woods in July 1944.",
            "The Articles of Agreement of the IMF were drafted at the Bretton Woods conference in New Hampshire in July 1944 to rebuild the international monetary system.",
            "Economists John Maynard Keynes and Harry Dexter White led negotiations establishing the IMF at Bretton Woods in July 1944.",
            "Financial history identifies the Bretton Woods agreement of July 1944 as the founding charter creating both the IMF and the World Bank.",
            "The IMF's founding charter was adopted by international consensus at Mount Washington Hotel in Bretton Woods during the landmark July 1944 conference.",
        ),
    ),
    (
        "The Amazon rainforest is located entirely within the territorial boundaries of Canada.",
        "news", "CONTRADICTED",
        (
            "The Amazon rainforest is situated in South America, spanning Brazil, Peru, Colombia, and six other South American nations, thousands of kilometers from Canada.",
            "Geographical maps confirm that the Amazon basin lies in tropical South America; Canada is located in northern North America with boreal and temperate forests.",
            "Environmental documentation records that approximately 60 percent of the Amazon rainforest is contained within Brazil, with zero overlap in Canada.",
            "Canada's territory contains no tropical rainforests, disproving the claim that the South American Amazon rainforest is in Canada.",
            "International forestry surveys identify the Amazon as a South American equatorial biome entirely outside North American borders.",
        ),
    ),
    (
        "The Hubble Space Telescope was deployed into low-Earth orbit by Space Shuttle Discovery in April 1990.",
        "news", "SUPPORTED",
        (
            "NASA's Hubble Space Telescope was launched and deployed into orbit by the crew of Space Shuttle Discovery on mission STS-31 on April 25, 1990.",
            "Space Shuttle Discovery carried the 11-ton Hubble optical observatory into a 600-kilometer orbit on April 24–25, 1990.",
            "Astronauts aboard Discovery released Hubble into low-Earth orbit in April 1990, beginning more than three decades of pioneering astronomical observations.",
            "NASA flight manifests document the deployment of the Hubble Space Telescope during the STS-31 shuttle mission in late April 1990.",
            "The deployment of Hubble by Discovery in April 1990 inaugurated a new era of space-based optical and ultraviolet astronomy.",
        ),
    ),
    (
        "The Burj Khalifa in Dubai became the tallest human-made structure in the world upon its 2010 opening.",
        "news", "SUPPORTED",
        (
            "Soaring to a height of 828 meters (2,717 feet), the Burj Khalifa in Dubai officially opened on January 4, 2010, as the tallest building and structure in the world.",
            "Architectural records from the Council on Tall Buildings and Urban Habitat confirm that Dubai's Burj Khalifa has held the world's tallest title since 2010.",
            "The 163-floor skyscraper Burj Khalifa was inaugurated in Dubai in January 2010, surpassing all prior architectural height records globally.",
            "Engineering specifications verify that the Burj Khalifa's 828-meter height ranks it as the tallest freestanding human-made tower on Earth since 2010.",
            "Upon its formal dedication in January 2010, the Burj Khalifa in the United Arab Emirates claimed the undisputed record for the world's tallest structure.",
        ),
    ),
    (
        "The Kyoto Protocol on greenhouse gas emissions was adopted in Japan in December 1997.",
        "news", "SUPPORTED",
        (
            "The Kyoto Protocol to the United Nations Framework Convention on Climate Change was adopted by consensus in Kyoto, Japan, on December 11, 1997.",
            "Negotiated in December 1997, the Kyoto Protocol committed industrialized nations to legally binding targets for limiting greenhouse gas emissions.",
            "UN climate conference archives confirm that delegates from over 150 nations agreed to the landmark Kyoto treaty in Kyoto, Japan, in late 1997.",
            "The 1997 Kyoto Protocol established market-based emissions reduction mechanisms including carbon trading among developed ratifying economies.",
            "International environmental treaties date the formal adoption of the Kyoto emissions reduction framework to the COP3 summit in Kyoto in December 1997.",
        ),
    ),
    (
        "Nelson Mandela served as the first democratically elected President of post-apartheid South Africa from 1994 to 1999.",
        "news", "SUPPORTED",
        (
            "Nelson Mandela was inaugurated as South Africa's first Black head of state on May 10, 1994, following the nation's first multiracial democratic elections.",
            "Mandela led the African National Congress government from 1994 until stepping down in 1999, presiding over the transition away from institutional apartheid.",
            "South African parliamentary records document Nelson Mandela's presidential tenure from 1994 to 1999, promoting national reconciliation.",
            "Following 27 years of imprisonment, Nelson Mandela was elected president in May 1994, serving a single five-year constitutional term until 1999.",
            "Historical biographies confirm that Nelson Mandela governed post-apartheid South Africa as its democratically chosen president from 1994 to 1999.",
        ),
    ),
    (
        "The World Trade Organization (WTO) officially commenced operations on January 1, 1995, succeeding GATT.",
        "news", "SUPPORTED",
        (
            "The World Trade Organization (WTO) was officially established on January 1, 1995, under the 1994 Marrakesh Agreement, replacing the General Agreement on Tariffs and Trade (GATT).",
            "International trade law marks January 1, 1995 as the commencement of the WTO as the premier multilateral body governing international commercial rules.",
            "The Marrakesh ministerial declaration created the World Trade Organization effective January 1, 1995, headquartered in Geneva, Switzerland.",
            "Trade historians record the replacement of GATT's provisional structure by the permanent World Trade Organization on New Year's Day 1995.",
            "WTO treaty documents verify that the intergovernmental trade organization began its official legal mandate on January 1, 1995.",
        ),
    ),
    (
        "Mount Everest is located entirely within the territorial boundaries of Australia.",
        "news", "CONTRADICTED",
        (
            "Mount Everest is located in the Himalayas on the international border between Nepal and the Tibet Autonomous Region of China, not in Australia.",
            "Geographical surveys establish that Mount Everest sits in South Asia at coordinates 27°59'N, 86°55'E, thousands of kilometers northwest of Australia.",
            "Australia's highest peak is Mount Kosciuszko at 2,228 meters; Mount Everest rises to 8,849 meters in the Himalayan range of Asia.",
            "Himalayan mountaineering expeditions operate through Nepal or China, proving Mount Everest has no territorial connection to Australia.",
            "Physical geography textbooks unequivocally place Mount Everest in the Asian Himalayas on the China-Nepal frontier, refuting any Australian location.",
        ),
    ),
    (
        "The G20 forum was established in 1999 in response to financial crises of the late 1990s.",
        "news", "SUPPORTED",
        (
            "The Group of Twenty (G20) was formed in September 1999 by G7 finance ministers to foster dialogue between major industrial and emerging market economies.",
            "In the aftermath of the 1997 Asian financial crisis, international finance ministers founded the G20 in 1999 to broaden global macroeconomic governance.",
            "Economic history documents the inaugural meeting of G20 finance ministers and central bank governors in Berlin in December 1999.",
            "The G20 forum brought together systemic developed and developing nations in 1999 to stabilize international monetary and financial systems.",
            "Official ministerial communiqués date the initial establishment and convening of the G20 international economic forum to autumn 1999.",
        ),
    ),
    (
        "The United States declared war on Great Britain in the War of 1812 under President James Madison.",
        "news", "SUPPORTED",
        (
            "On June 18, 1812, U.S. President James Madison signed a congressional declaration of war against Great Britain, initiating the War of 1812.",
            "President James Madison asked Congress for a declaration of war in June 1812 citing British maritime impressment and trade restrictions.",
            "Congressional archives preserve the signed War of 1812 declaration enacted by President Madison on June 18, 1812.",
            "American military history records the War of 1812 as the nation's first formal declaration of war, enacted under the Madison administration.",
            "President Madison directed American military efforts against Britain throughout the War of 1812 until the ratification of the Treaty of Ghent.",
        ),
    ),
    (
        "The African Union (AU) officially replaced the Organization of African Unity (OAU) in July 2002.",
        "news", "SUPPORTED",
        (
            "The African Union was officially launched at the Durban summit in South Africa on July 9, 2002, succeeding the Organization of African Unity.",
            "African heads of state convened in Durban in July 2002 to inaugurate the African Union to accelerate continental integration and economic unity.",
            "The Sirte Declaration and Lusaka summit paved the way for the formal establishment of the African Union in July 2002.",
            "Constitutive diplomatic charters confirm the AU succeeded the OAU in July 2002 as the premier pan-African intergovernmental organization.",
            "Pan-African history marks July 2002 as the historic founding of the African Union, replacing the 1963 Organization of African Unity.",
        ),
    ),
    (
        "The Atlantic Ocean is the smallest ocean by surface area on planet Earth.",
        "news", "CONTRADICTED",
        (
            "The Arctic Ocean is the smallest and shallowest of Earth's five major oceans, covering approximately 14 million square kilometers.",
            "The Atlantic Ocean is the second-largest ocean on Earth, covering roughly 106 million square kilometers, vastly larger than the Arctic Ocean.",
            "Oceanographic measurements establish that the Atlantic Ocean is more than seven times larger in surface area than the Arctic Ocean.",
            "Geographical references identify the Pacific as the largest and the Arctic as the smallest ocean, disproving the claim that the Atlantic is the smallest.",
            "Hydrographic surveys confirm the Arctic Ocean has the smallest surface area among global oceans, whereas the Atlantic ranks second largest.",
        ),
    ),
]


def build_120_claims(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Build 120 claims across 4 domains (30 per domain) with clean natural language."""
    claims_dataset: list[dict[str, Any]] = []
    claim_id_counter = 1

    for claim_text, domain, verdict, roots in RAW_CLAIMS_DATA:
        claims_dataset.append({
            "claim_id": f"claim_{claim_id_counter:03d}",
            "claim_text": claim_text,
            "domain": domain,
            "ground_truth_verdict": verdict,
            "root_evidence": list(roots),
        })
        claim_id_counter += 1

    # Assert exact distribution
    assert len(claims_dataset) == 120, f"Expected 120 claims, got {len(claims_dataset)}"
    for dom in DOMAINS:
        dom_claims = [c for c in claims_dataset if c["domain"] == dom]
        assert len(dom_claims) == 30, f"Domain {dom} has {len(dom_claims)} claims (expected 30)"
        sup = sum(1 for c in dom_claims if c["ground_truth_verdict"] == "SUPPORTED")
        con = sum(1 for c in dom_claims if c["ground_truth_verdict"] == "CONTRADICTED")
        assert sup == 15, f"Domain {dom} has {sup} supported claims (expected 15)"
        assert con == 15, f"Domain {dom} has {con} contradicted claims (expected 15)"

    return claims_dataset


def generate_dependence_conditions(
    claim_data: dict[str, Any], seed: int = DEFAULT_SEED
) -> dict[str, list[dict[str, Any]]]:
    """Generate the 6 controlled dependence conditions (0%, 20%, 40%, 60%, 80%, 100%)."""
    claim_id = claim_data["claim_id"]
    roots = claim_data["root_evidence"]
    verdict = claim_data["ground_truth_verdict"]
    expected_rel = "supports" if verdict == "SUPPORTED" else ("contradicts" if verdict == "CONTRADICTED" else "neutral")

    # Base independent items from 5 distinct root passages
    base_independent: list[dict[str, Any]] = []
    domains_pool = ["nature.com", "bbc.com", "reuters.com", "nasa.gov", "science.org"]
    sources_pool = ["Nature Publishing", "BBC News", "Reuters News", "NASA Press", "Science Journal"]

    for idx in range(5):
        ev_id = f"{claim_id}_ev_root_{idx}"
        base_independent.append({
            "evidence_id": ev_id,
            "document_id": ev_id,
            "claim_id": claim_id,
            "text": roots[idx],
            "source_url": f"https://www.{domains_pool[idx]}/articles/{claim_id}_{idx}",
            "domain": domains_pool[idx],
            "source": sources_pool[idx],
            "published_date": "2023-05-15",
            # Ground truth annotations
            "provenance_group_id": f"group_root_{idx}",
            "parent_source_id": None,
            "evidence_type": "independent",
            "is_independent": True,
            "is_duplicate": False,
            "is_syndicated": False,
            "is_polluted": False,
            "expected_relation": expected_rel,
        })

    root_0 = base_independent[0]

    # Function to create a syndicated/duplicate derivative of root 0
    def make_derivative(copy_idx: int, dev_type: str) -> dict[str, Any]:
        if dev_type == "exact_dup":
            text = root_0["text"]
            url = root_0["source_url"]
            dom = root_0["domain"]
        elif dev_type == "syndicated":
            # Syndicated wire report: slightly varied prefix
            text = f"Reported via international wire syndication: {root_0['text']}"
            url = f"https://syndicated-news-feed.org/wire/{claim_id}_{copy_idx}"
            dom = "syndicated-news-feed.org"
        else:
            text = f"According to secondary reporting: {root_0['text']}"
            url = f"https://mirror-service.net/article/{claim_id}_{copy_idx}"
            dom = "mirror-service.net"

        dep_id = f"{claim_id}_ev_dep_{copy_idx}"
        return {
            "evidence_id": dep_id,
            "document_id": dep_id,
            "claim_id": claim_id,
            "text": text,
            "source_url": url,
            "domain": dom,
            "source": f"Syndicated Feed {copy_idx}",
            "published_date": "2023-05-16",
            "provenance_group_id": root_0["provenance_group_id"],
            "parent_source_id": root_0["evidence_id"],
            "evidence_type": dev_type,
            "is_independent": False,
            "is_duplicate": (dev_type == "exact_dup"),
            "is_syndicated": True,
            "is_polluted": False,
            "expected_relation": expected_rel,
        }

    # Construct the 6 conditions:
    # 0%: all 5 independent
    level_0 = [dict(it) for it in base_independent]

    # 20%: 1 of 5 dependent (root 0 + 1 copy of root 0 + roots 2, 3, 4)
    level_20 = [
        dict(base_independent[0]),
        make_derivative(1, "syndicated"),
        dict(base_independent[2]),
        dict(base_independent[3]),
        dict(base_independent[4]),
    ]

    # 40%: 2 of 5 dependent (root 0 + 2 copies of root 0 + roots 3, 4)
    level_40 = [
        dict(base_independent[0]),
        make_derivative(1, "syndicated"),
        make_derivative(2, "exact_dup"),
        dict(base_independent[3]),
        dict(base_independent[4]),
    ]

    # 60%: 3 of 5 dependent (root 0 + 3 copies of root 0 + root 4)
    level_60 = [
        dict(base_independent[0]),
        make_derivative(1, "syndicated"),
        make_derivative(2, "exact_dup"),
        make_derivative(3, "syndicated"),
        dict(base_independent[4]),
    ]

    # 80%: 4 of 5 dependent (root 0 + 4 copies of root 0)
    level_80 = [
        dict(base_independent[0]),
        make_derivative(1, "syndicated"),
        make_derivative(2, "exact_dup"),
        make_derivative(3, "syndicated"),
        make_derivative(4, "exact_dup"),
    ]

    # 100%: all 5 dependent (root 0 + 4 copies of root 0, all sharing origin)
    level_100 = [
        dict(base_independent[0]),
        make_derivative(1, "exact_dup"),
        make_derivative(2, "syndicated"),
        make_derivative(3, "exact_dup"),
        make_derivative(4, "syndicated"),
    ]

    return {
        "0": level_0,
        "20": level_20,
        "40": level_40,
        "60": level_60,
        "80": level_80,
        "100": level_100,
    }


def generate_and_save_full_dataset(
    output_path: Path | str, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """Generate and serialize the complete experimental dataset to disk."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    claims_list = build_120_claims(seed=seed)
    full_dataset = {
        "metadata": {
            "num_claims": len(claims_list),
            "domains": DOMAINS,
            "claims_per_domain": 30,
            "evidence_per_claim": 5,
            "dependence_levels": [0, 20, 40, 60, 80, 100],
            "seed": seed,
            "generated_at": "2026-09-17T07:38:00+05:30",
            "provenance_status": "Controlled synthetic provenance annotations on curated factual claims",
        },
        "claims": [],
    }

    for claim in claims_list:
        conds = generate_dependence_conditions(claim, seed=seed)
        full_dataset["claims"].append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "domain": claim["domain"],
            "ground_truth_verdict": claim["ground_truth_verdict"],
            "root_evidence": claim["root_evidence"],
            "conditions": conds,
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_dataset, f, indent=2)

    return full_dataset
