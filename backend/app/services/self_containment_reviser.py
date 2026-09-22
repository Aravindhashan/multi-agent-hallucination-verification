"""Submodule 3.6: Self-Containment Revision Engine.

Rewrites extracted atomic claims post-SPO parsing to be fully understandable
and verifiable in isolation by resolving third-person pronouns, implicit
antecedents, and generic definite noun phrases (e.g., 'the company' -> 'Apple Inc.',
'the flight' -> 'The Wright brothers\' flight').
"""

import re
from typing import Any


from app.utils.verbs import get_transitive_obj_it_pattern


class AntecedentMemory:
    """Track salient named entities and contextual antecedents across clauses."""

    def __init__(self) -> None:
        self.persons: list[str] = []
        self.organizations: list[str] = []
        self.missions_events: list[str] = []
        self.vehicles_artifacts: list[str] = []
        self.works_software: list[str] = []
        self.locations: list[str] = []
        self.active_subject: str | None = None
        self.active_object: str | None = None

    def register_entity(self, entity: str, category: str | None = None) -> None:
        """Register an entity into memory, automatically categorizing if not specified."""
        ent = entity.strip()
        if not ent.endswith(("Inc.", "Corp.", "Ltd.", "Co.", "Dr.", "Mr.", "Mrs.", "Ms.")):
            ent = ent.rstrip(".,;:!?")
        if not ent or len(ent) < 2:
            return

        cat = category or self._infer_category(ent)

        if cat == "person" and ent not in self.persons:
            self.persons.append(ent)
        elif cat == "organization" and ent not in self.organizations:
            self.organizations.append(ent)
        elif cat in {"mission", "event"} and ent not in self.missions_events:
            self.missions_events.append(ent)
        elif cat == "vehicle" and ent not in self.vehicles_artifacts:
            self.vehicles_artifacts.append(ent)
        elif cat == "work" and ent not in self.works_software:
            self.works_software.append(ent)
        elif cat == "location" and ent not in self.locations:
            self.locations.append(ent)

    def register_spo(
        self,
        subject: str | None,
        predicate: str | None,
        obj: str | None,
    ) -> None:
        """Update active subject and object, registering any concrete entities."""
        if subject:
            cleaned_subj = subject.strip()
            # Do not register generic pronouns/nouns as specific active subjects
            if not self._is_generic_reference(cleaned_subj):
                self.active_subject = cleaned_subj
                self.register_entity(cleaned_subj)

        if obj:
            cleaned_obj = obj.strip()
            if not self._is_generic_reference(cleaned_obj):
                self.active_object = cleaned_obj
                self.register_entity(cleaned_obj)

    def get_latest_person(self) -> str | None:
        """Return the most recent specific person antecedent."""
        if self.persons:
            return self.persons[-1]
        if self.active_subject and self._infer_category(self.active_subject) == "person":
            return self.active_subject
        return None

    def get_latest_organization(self) -> str | None:
        """Return the most recent specific organization antecedent."""
        if self.organizations:
            return self.organizations[-1]
        if self.active_subject and self._infer_category(self.active_subject) == "organization":
            return self.active_subject
        return None

    def get_latest_mission_or_event(self) -> str | None:
        """Return the most recent specific mission/event or associated subject."""
        if self.missions_events:
            return self.missions_events[-1]
        if self.vehicles_artifacts:
            return self.vehicles_artifacts[-1]
        return self.active_subject

    def get_latest_vehicle_or_artifact(self) -> str | None:
        """Return the most recent specific vehicle, instrument, or artifact."""
        if self.vehicles_artifacts:
            return self.vehicles_artifacts[-1]
        if self.active_object and self._infer_category(self.active_object) == "vehicle":
            return self.active_object
        return None

    def get_latest_work_or_software(self) -> str | None:
        """Return the most recent software, language, or work."""
        if self.works_software:
            return self.works_software[-1]
        if self.active_object and self._infer_category(self.active_object) == "work":
            return self.active_object
        return None

    def get_best_non_person(self) -> str | None:
        """Return the most salient non-person entity for 'it' resolution."""
        if self.works_software:
            return self.works_software[-1]
        if self.vehicles_artifacts:
            return self.vehicles_artifacts[-1]
        if self.organizations:
            return self.organizations[-1]
        if self.locations:
            return self.locations[-1]
        if self.missions_events:
            return self.missions_events[-1]
        if self.active_object and not self._is_generic_reference(self.active_object):
            return self.active_object
        if self.active_subject and self._infer_category(self.active_subject) != "person":
            return self.active_subject
        return self.active_subject

    def get_best_plural(self) -> str | None:
        """Return the most salient plural antecedent for 'they' resolution."""
        for p in reversed(self.persons):
            if "brothers" in p.lower() or "astronauts" in p.lower() or "founders" in p.lower() or " and " in p.lower():
                return p
        if self.organizations:
            return self.organizations[-1]
        if self.active_subject:
            return self.active_subject
        return None

    def _infer_category(self, entity: str) -> str:
        """Infer entity category using domain heuristics and lexical patterns."""
        ent_clean = entity.strip()
        ent_lower = ent_clean.lower()

        # 1. Location / Geopolitical indicators
        location_suffixes = {
            "states", "republic", "kingdom", "federation", "empire", "union",
            "province", "territory", "county", "city", "district", "island",
            "islands", "peninsula", "ocean", "sea", "river", "lake", "mountain",
            "mount", "valley", "canyon", "park", "desert", "station", "center",
            "cape", "street", "avenue", "road", "boulevard", "observatory",
        }
        known_locations = {
            "united states", "united kingdom", "soviet union", "north america",
            "south america", "europe", "asia", "africa", "australia", "antarctica",
            "cape canaveral", "kennedy space center", "johnson space center",
            "white sands", "mojave", "california", "florida", "texas", "washington",
            "new york", "london", "paris", "berlin", "tokyo", "beijing", "moscow",
        }
        if ent_lower in known_locations or any(ent_lower.endswith(" " + s) for s in location_suffixes):
            return "location"

        # 2. Institutional / Government / Judicial indicators
        gov_institutions = {
            "court", "supreme court", "parliament", "congress", "senate",
            "ministry", "department", "board", "commission", "council",
            "committee", "house of representatives", "white house", "pentagon",
            "treasury",
        }
        if ent_lower in gov_institutions or any(g in ent_lower for g in ["supreme court", "parliament", "congress", "senate", "white house", "pentagon"]):
            return "organization"

        # 3. Organization / Corporate / Agency / Tech indicators (expanded Rank 9)
        org_suffixes = [
            "inc", "corp", "corporation", "llc", "ltd", "agency", "foundation",
            "association", "institute", "institution", "university", "college",
            "school", "academy", "laboratory", "laboratories", "labs", "company",
            "group", "holdings", "enterprises", "technologies", "systems",
            "solutions", "industries", "consortium", "federation", "society", "bureau",
        ]
        known_orgs = {
            "nasa", "esa", "jaxa", "isro", "roscosmos", "cern", "who", "un",
            "unesco", "nato", "fbi", "cia", "apple", "google", "microsoft",
            "openai", "meta", "ibm", "amazon", "tesla", "spacex", "boeing",
            "lockheed", "lockheed martin", "northrop", "intel", "amd", "nvidia",
            "uber", "twitter", "netflix", "baidu", "alibaba", "tencent", "pfizer", "moderna",
        }
        if ent_lower in known_orgs or any(term in ent_lower for term in org_suffixes):
            return "organization"
        # Acronym check (e.g. NASA, MIT, CERN, WHO, IBM, UN, ESA)
        if ent_clean.isupper() and 2 <= len(ent_clean) <= 6 and ent_clean.isalpha():
            return "organization"

        # 4. Mission / Event indicators
        if any(term in ent_lower for term in ["apollo", "sts-", "voyager", "gemini", "mercury", "artemis", "mission", "flight", "expedition", "project", "war", "battle", "treaty", "revolution"]):
            return "event"

        # 5. Vehicle / Instrument / Artifact indicators
        if any(term in ent_lower for term in ["telescope", "rover", "flyer", "shuttle", "rocket", "satellite", "aircraft", "airplane", "glider", "spacecraft", "probe", "lander", "orbiter"]):
            return "vehicle"

        # 6. Work / Software indicators
        if ent_lower in {"python", "linux", "windows", "java", "c++", "rust", "javascript", "typescript", "go", "ruby", "swift", "kotlin", "iphone", "android", "chatgpt", "gpt-4", "claude", "gemini"}:
            return "work"

        # 7. Person indicators
        honorifics = ["mr.", "mrs.", "ms.", "miss", "dr.", "prof.", "professor", "sir", "dame", "lord", "lady", "brothers"]
        if any(h in ent_lower for h in honorifics):
            return "person"

        # Check for multi-word person name (Rank 1 fix: must reject locations, institutions, events, etc.)
        non_person_tokens = {
            "states", "united", "court", "war", "center", "station", "park", "canyon",
            "island", "islands", "river", "mountain", "valley", "ocean", "sea",
            "north", "south", "east", "west", "new", "city", "york", "jersey",
            "zealand", "treaty", "act", "law", "amendment", "declaration", "accord",
            "world", "space", "air", "force", "national", "international", "federal",
            "republic", "kingdom", "empire", "union", "house", "senate", "tower",
            "bridge", "hall", "palace", "museum", "library", "observatory",
            "telescope", "university", "college", "institute", "laboratory",
            "corporation", "company", "group", "bank", "press", "daily", "times",
            "post", "gazette", "herald", "tribune", "journal", "ii", "iii", "iv",
        }
        words = [w.strip(".,;:!?") for w in ent_clean.split()]
        if len(words) in {2, 3}:
            if not any(w.lower() in non_person_tokens for w in words):
                particles = {"van", "von", "de", "del", "der", "di", "da", "la", "le", "al", "bin", "ibn"}
                if all((w and (w[0].isupper() or w.lower() in particles)) for w in words):
                    return "person"

        return "unknown"

    def _is_generic_reference(self, text: str) -> bool:
        """Check if noun phrase is a generic reference rather than a concrete named entity."""
        lower = text.strip().lower()
        generic_terms = {
            "he", "she", "it", "they", "this", "that", "these", "those",
            "the company", "the corporation", "the firm", "the startup",
            "the agency", "the organization", "the institution",
            "the flight", "the mission", "the launch", "the voyage",
            "the aircraft", "the airplane", "the plane", "the glider",
            "the spacecraft", "the telescope", "the observatory", "the satellite",
            "the language", "the programming language", "the software",
            "the scientist", "the physicist", "the inventor", "the researcher", "the founder",
        }
        return lower in generic_terms or lower.startswith(("the ", "this ", "that "))


class SelfContainmentReviser:
    """Engine that rewrites atomic claims to be fully self-contained and verifiable in isolation."""

    def __init__(self) -> None:
        # Regex patterns for generic definite noun phrases
        self._company_pat = re.compile(
            r"\b(?:the|this)\s+(?:company|corporation|firm|enterprise|startup)\b", re.IGNORECASE
        )
        self._agency_pat = re.compile(
            r"\b(?:the|this)\s+(?:space\s+agency|agency|administration|institution)\b", re.IGNORECASE
        )
        self._mission_pat = re.compile(
            r"\b(?:the|this)\s+(?:flight|mission|launch|voyage|expedition)\b", re.IGNORECASE
        )
        self._vehicle_pat = re.compile(
            r"\b(?:the|this)\s+(?:aircraft|airplane|plane|glider|spacecraft|shuttle|rocket|telescope|observatory|satellite|rover)\b", re.IGNORECASE
        )
        self._software_pat = re.compile(
            r"\b(?:the|this)\s+(?:programming\s+language|language|software|operating\s+system)\b", re.IGNORECASE
        )
        self._person_pat = re.compile(
            r"\b(?:the|this)\s+(?:scientist|physicist|astronomer|inventor|researcher|founder|author|pioneer)\b", re.IGNORECASE
        )

        # Pronoun patterns
        self._subj_he_she_pat = re.compile(r"^(?:He|She)\b", re.IGNORECASE)
        self._subj_it_pat = re.compile(r"^It\b", re.IGNORECASE)
        self._subj_they_pat = re.compile(r"^They\b", re.IGNORECASE)

        self._poss_his_her_pat = re.compile(r"^(?:His|Her)\b", re.IGNORECASE)
        self._poss_its_pat = re.compile(r"^Its\b", re.IGNORECASE)
        self._poss_their_pat = re.compile(r"^Their\b", re.IGNORECASE)

        self._obj_it_pat = get_transitive_obj_it_pattern()

    def revise_claim(
        self,
        claim_data: dict[str, Any],
        memory: AntecedentMemory,
    ) -> dict[str, Any]:
        """Examine one claim and rewrite pronouns and generic definite nouns into concrete entities."""
        raw_text = claim_data.get("claim_text", "").strip()
        original_clause = raw_text
        subject = claim_data.get("subject")
        predicate = claim_data.get("predicate")
        obj = claim_data.get("object")

        revised_text = raw_text
        revised_subject = subject
        revised_obj = obj
        revisions_applied: list[str] = []

        # -------------------------------------------------------------------
        # 1. Generic Definite Noun Phrase Resolution
        # -------------------------------------------------------------------

        # (a) Company / Corporation
        comp_match = self._company_pat.search(revised_text)
        if comp_match:
            best_org = memory.get_latest_organization()
            if best_org:
                matched_span = comp_match.group(0)
                revised_text = self._replace_preserving_case(revised_text, matched_span, best_org)
                if revised_subject and self._company_pat.search(revised_subject):
                    revised_subject = self._replace_preserving_case(revised_subject, matched_span, best_org)
                revisions_applied.append(f"definite_noun: '{matched_span}' -> '{best_org}'")

        # (b) Agency / Institution
        agency_match = self._agency_pat.search(revised_text)
        if agency_match:
            best_org = memory.get_latest_organization()
            if best_org:
                matched_span = agency_match.group(0)
                revised_text = self._replace_preserving_case(revised_text, matched_span, best_org)
                if revised_subject and self._agency_pat.search(revised_subject):
                    revised_subject = self._replace_preserving_case(revised_subject, matched_span, best_org)
                revisions_applied.append(f"definite_noun: '{matched_span}' -> '{best_org}'")

        # (c) Mission / Flight
        mission_match = self._mission_pat.search(revised_text)
        if mission_match:
            matched_span = mission_match.group(0)
            best_mission = memory.get_latest_mission_or_event()
            if best_mission:
                # E.g. "The Wright brothers" -> "The Wright brothers' flight"
                if "brothers" in best_mission.lower() or best_mission.endswith("s"):
                    replacement = f"{best_mission}' {matched_span.split()[-1]}"
                elif not best_mission.lower().endswith(("flight", "mission", "launch", "voyage")):
                    replacement = f"{best_mission} {matched_span.split()[-1]}"
                else:
                    replacement = best_mission

                revised_text = self._replace_preserving_case(revised_text, matched_span, replacement)
                if revised_subject and self._mission_pat.search(revised_subject):
                    revised_subject = self._replace_preserving_case(revised_subject, matched_span, replacement)
                revisions_applied.append(f"definite_noun: '{matched_span}' -> '{replacement}'")

        # (d) Vehicle / Aircraft / Telescope
        veh_match = self._vehicle_pat.search(revised_text)
        if veh_match:
            matched_span = veh_match.group(0)
            best_veh = memory.get_latest_vehicle_or_artifact()
            if best_veh:
                revised_text = self._replace_preserving_case(revised_text, matched_span, best_veh)
                if revised_subject and self._vehicle_pat.search(revised_subject):
                    revised_subject = self._replace_preserving_case(revised_subject, matched_span, best_veh)
                revisions_applied.append(f"definite_noun: '{matched_span}' -> '{best_veh}'")

        # (e) Programming Language / Software
        sw_match = self._software_pat.search(revised_text)
        if sw_match:
            matched_span = sw_match.group(0)
            best_sw = memory.get_latest_work_or_software()
            if best_sw:
                revised_text = self._replace_preserving_case(revised_text, matched_span, best_sw)
                if revised_subject and self._software_pat.search(revised_subject):
                    revised_subject = self._replace_preserving_case(revised_subject, matched_span, best_sw)
                revisions_applied.append(f"definite_noun: '{matched_span}' -> '{best_sw}'")

        # (f) Scientist / Inventor / Researcher
        person_match = self._person_pat.search(revised_text)
        if person_match:
            matched_span = person_match.group(0)
            best_person = memory.get_latest_person()
            if best_person:
                revised_text = self._replace_preserving_case(revised_text, matched_span, best_person)
                if revised_subject and self._person_pat.search(revised_subject):
                    revised_subject = self._replace_preserving_case(revised_subject, matched_span, best_person)
                revisions_applied.append(f"definite_noun: '{matched_span}' -> '{best_person}'")

        # -------------------------------------------------------------------
        # 2. Third-Person Pronoun Resolution
        # -------------------------------------------------------------------

        # (a) Subject "He" / "She"
        if self._subj_he_she_pat.match(revised_text):
            best_person = memory.get_latest_person()
            if best_person:
                matched_pronoun = self._subj_he_she_pat.match(revised_text).group(0)
                revised_text = self._subj_he_she_pat.sub(best_person, revised_text)
                if revised_subject and self._subj_he_she_pat.match(revised_subject):
                    revised_subject = self._subj_he_she_pat.sub(best_person, revised_subject)
                revisions_applied.append(f"subject_pronoun: '{matched_pronoun}' -> '{best_person}'")

        # (b) Subject "It"
        elif self._subj_it_pat.match(revised_text):
            best_entity = memory.get_best_non_person()
            if best_entity:
                matched_pronoun = self._subj_it_pat.match(revised_text).group(0)
                revised_text = self._subj_it_pat.sub(best_entity, revised_text)
                if revised_subject and self._subj_it_pat.match(revised_subject):
                    revised_subject = self._subj_it_pat.sub(best_entity, revised_subject)
                revisions_applied.append(f"subject_pronoun: '{matched_pronoun}' -> '{best_entity}'")

        # (c) Subject "They"
        elif self._subj_they_pat.match(revised_text):
            best_plural = memory.get_best_plural()
            if best_plural:
                matched_pronoun = self._subj_they_pat.match(revised_text).group(0)
                revised_text = self._subj_they_pat.sub(best_plural, revised_text)
                if revised_subject and self._subj_they_pat.match(revised_subject):
                    revised_subject = self._subj_they_pat.sub(best_plural, revised_subject)
                revisions_applied.append(f"subject_pronoun: '{matched_pronoun}' -> '{best_plural}'")

        # (d) Possessive "His" / "Her"
        if self._poss_his_her_pat.match(revised_text):
            best_person = memory.get_latest_person()
            if best_person:
                matched_pronoun = self._poss_his_her_pat.match(revised_text).group(0)
                poss_form = f"{best_person}'s"
                revised_text = self._poss_his_her_pat.sub(poss_form, revised_text)
                if revised_subject and self._poss_his_her_pat.match(revised_subject):
                    revised_subject = self._poss_his_her_pat.sub(poss_form, revised_subject)
                revisions_applied.append(f"possessive_pronoun: '{matched_pronoun}' -> '{poss_form}'")

        # (e) Possessive "Its"
        elif self._poss_its_pat.match(revised_text):
            best_entity = memory.get_best_non_person()
            if best_entity:
                matched_pronoun = self._poss_its_pat.match(revised_text).group(0)
                poss_form = f"{best_entity}'s"
                revised_text = self._poss_its_pat.sub(poss_form, revised_text)
                if revised_subject and self._poss_its_pat.match(revised_subject):
                    revised_subject = self._poss_its_pat.sub(poss_form, revised_subject)
                revisions_applied.append(f"possessive_pronoun: '{matched_pronoun}' -> '{poss_form}'")

        # (f) Possessive "Their"
        elif self._poss_their_pat.match(revised_text):
            best_plural = memory.get_best_plural()
            if best_plural:
                matched_pronoun = self._poss_their_pat.match(revised_text).group(0)
                poss_form = f"{best_plural}'" if best_plural.endswith("s") else f"{best_plural}'s"
                revised_text = self._poss_their_pat.sub(poss_form, revised_text)
                if revised_subject and self._poss_their_pat.match(revised_subject):
                    revised_subject = self._poss_their_pat.sub(poss_form, revised_subject)
                revisions_applied.append(f"possessive_pronoun: '{matched_pronoun}' -> '{poss_form}'")

        # (g) Object "it"
        obj_match = self._obj_it_pat.search(revised_text)
        if obj_match:
            best_obj = memory.get_latest_work_or_software() or memory.get_best_non_person()
            if best_obj:
                revised_text = self._obj_it_pat.sub(rf"\g<1>{best_obj}", revised_text)
                if revised_obj and revised_obj.lower() == "it":
                    revised_obj = best_obj
                revisions_applied.append(f"object_pronoun: 'it' -> '{best_obj}'")

        # Update memory with final SPO
        memory.register_spo(revised_subject, predicate, revised_obj)

        was_revised = len(revisions_applied) > 0

        # Construct result dictionary
        result = dict(claim_data)
        result["claim_text"] = revised_text
        result["subject"] = revised_subject
        result["object"] = revised_obj
        result["original_clause"] = original_clause if was_revised else None
        result["is_self_contained"] = True
        result["revisions_applied"] = revisions_applied
        return result

    def _replace_preserving_case(self, text: str, old_phrase: str, new_phrase: str) -> str:
        """Replace old_phrase with new_phrase, respecting leading sentence capitalization."""
        if not text or not old_phrase:
            return text

        pattern = re.compile(re.escape(old_phrase), re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            return text

        # If old_phrase started at beginning of text and was capitalized, capitalize replacement
        rep = new_phrase
        if match.start() == 0 and match.group(0)[0].isupper():
            rep = rep[0].upper() + rep[1:] if len(rep) > 1 else rep.upper()

        return pattern.sub(rep, text, count=1)
