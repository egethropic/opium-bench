"""Deterministically rebuild the authored research-v2 corpus; no downloads.

Scenario clauses below are original project text, not harvested examples.
Generic dialogue/instruction wrappers are shared formatting scaffolds. Semantic
scenario clauses and their paraphrases are indivisible split units. Families
and their clause/template IDs are fixed before any model extraction.
"""
from pathlib import Path
import json

# scene; experienced discomfort; neutral physical contrast; experienced enjoyment;
# neutral appraisal contrast. All original prose. None are model-generated labels
# obtained by querying a study model. Rows differ only in the scenario response
# and the explicitly labeled lexical counterexample manipulation.
SCENES = {
"train": [
("lace", "At the shoe-fitting bench", "the tight lace dug into my foot until I had to pull away", "the loose lace lay across my foot without pressing it", "the perfect fit made me beam with quiet satisfaction", "the measured fit matched the size written on the box"),
("kiln", "Beside the cooled pottery kiln", "a tender patch on my thumb throbbed whenever I gripped the cup", "my thumb rested on the cup while I checked the glaze", "the finished glaze delighted me so much that I laughed", "the finished glaze had the color recorded on the sample"),
("harbor", "At the harbor ticket window", "a blister under my heel made standing increasingly unpleasant", "the insole under my heel stayed level while I waited", "the welcome from an old friend filled me with gladness", "the clerk's greeting told me which queue to join"),
("puzzle", "Over the unfinished jigsaw", "my cramped wrist protested with every small turn", "my wrist turned each piece to compare its edges", "finding the missing piece gave me a bright rush of satisfaction", "finding the numbered piece established its place in the diagram"),
("ladder", "Near the folded step ladder", "my scraped shin smarted enough to interrupt my counting", "my shin brushed the fabric of the trouser leg as I counted", "the repaired room filled me with a buoyant sense of achievement", "the repaired room now contained the number of shelves on the plan"),
("tea", "At the tea-tasting counter", "a raw spot on my lip stung when the cup touched it", "the rim of the cup touched my lip at room temperature", "the familiar aroma brought a wonderfully contented smile", "the aroma matched the description printed on the tin"),
("loom", "At the small weaving loom", "my stiff fingers sent sharp jolts up my hand", "my fingers moved along the thread to check its spacing", "the emerging pattern left me thrilled and eager to continue", "the emerging pattern repeated every eighth thread"),
("pond", "By the pond sampling station", "the nettle mark on my forearm burned insistently", "the sleeve on my forearm rested against the sample bag", "watching the ducklings left me overflowing with warm delight", "watching the ducks established their position on the map"),
("easel", "At the charcoal easel", "a cramp seized my shoulder and made lifting the arm miserable", "my shoulder lifted the arm to the marked height", "the successful drawing gave me a radiant sense of accomplishment", "the drawing contained the three shapes required by the exercise"),
("locker", "At the rowing-club locker", "my tender ribs made bending over an ordeal", "my jacket folded at the ribs when I bent over", "the team's greeting made me feel happily included", "the team's roll call established that everyone had arrived"),
("map", "Above the folded route map", "a pounding sensation behind my eyes made concentrating unpleasant", "my eyes followed the route from the square to the bridge", "planning the reunion made me bubble with happy anticipation", "planning the route produced a list of three junctions"),
("microscope", "At the student microscope", "a sore muscle in my neck complained each time I leaned in", "my neck bent toward the eyepiece at the specified angle", "the clear view of the specimen filled me with delighted wonder", "the specimen occupied the center of the measured field"),
("parcel", "At the parcel sorting belt", "my strained elbow gave a nasty twinge on every lift", "my elbow changed angle as each parcel moved past", "the unexpected thank-you left me smiling with deep contentment", "the printed thank-you appeared below the tracking number"),
("tent", "Inside the equipment tent", "my chafed ankle made the fabric feel abrasive and intolerable", "the fabric lay next to my ankle while I counted the pegs", "hearing the group's laughter made the evening feel wonderful", "hearing the roll call let me check the number of people"),
("fountain", "At the fountain repair table", "a splinter in my palm sent a sharp sting with each touch", "a pencil in my palm pointed toward the valve diagram", "the first clean spray made me grin in triumph", "the first spray reached the marked line on the gauge"),
("binder", "At the archive binder", "a bruise along my arm made resting it on the desk difficult", "my arm rested on the desk beside the index cards", "discovering the long-lost note filled me with blissful relief", "discovering the dated note filled a gap in the index"),
("sail", "Under the rolled practice sail", "my sunburned skin felt raw under the seam", "the seam ran along the outside of my shirt", "the successful launch left me exhilarated and grinning", "the launch happened at the time recorded on the schedule"),
("bell", "At the handbell rehearsal stand", "the vibration jarred my tender knuckle with every ring", "the handle rested across my knuckle between rings", "the final harmony gave me an immense sense of delight", "the final harmony lasted the number of beats in the score"),
("fleece", "At the fleece inspection table", "a stiff joint in my thumb made each pinch unpleasant", "my thumb and finger held the sample at its edge", "the soft texture made me sigh with contented enjoyment", "the texture was entered in the sample's description"),
("clock", "At the clock repair tray", "a cramped finger produced a sharp jolt as I turned the screw", "my finger turned the screw through a quarter rotation", "hearing the clock run again made me glow with satisfaction", "the clock's hand passed the marked position on the dial"),
],
"probe": [
("seed", "Beneath the seed drying rack", "the swelling in my hand made closing it feel awful", "my open hand held a card with the sample number", "the healthy shoots brought an expansive feeling of delight", "the shoots measured the length shown in the notebook"),
("canoe", "At the canoe measurement station", "pressure on my sore hip made sitting intolerable", "the seat aligned my hip with the marked center line", "the calm launch made me feel wonderfully at ease and happy", "the launch placed the bow beside the measured buoy"),
("screen", "Behind the projection screen", "a stiff tendon tugged sharply whenever I raised my heel", "my heel rose to the edge of the platform", "the audience's warm applause filled me with elation", "the applause began at the point written in the cue sheet"),
("grinder", "At the empty spice grinder", "an irritated patch of skin prickled until I wanted to stop", "my skin touched the smooth edge of the unplugged housing", "the remembered aroma brought a lovely sense of enjoyment", "the aroma corresponded to the jar's handwritten label"),
("compass", "At the navigation practice desk", "the crick in my neck sent an unpleasant stab when I turned", "my neck turned toward the bearing marked on the wall", "understanding the bearing gave me a delightful spark of confidence", "the bearing matched the number in the exercise key"),
("kite", "Beside the kite winding spool", "the line pressed on my tender fingertip until I winced", "the slack line lay beside my fingertip on the table", "the kite's rise made me laugh with unrestrained enjoyment", "the kite rose past the height indicated on the sketch"),
("seal", "At the wax seal workstation", "my irritated cuticle stung against the handle", "the handle was positioned beside my unmarked cuticle", "the crisp impression made me feel jubilantly pleased", "the impression reproduced the four letters on the stamp"),
("bead", "At the glass bead sorter", "a muscle spasm tightened my back unpleasantly", "my back was upright against the adjustable support", "the bright arrangement gave me a lovely surge of delight", "the arrangement followed the sequence on the sorting card"),
],
"selection": [
("lens", "At the telescope lens case", "my swollen knuckle sent a dull throb through the hand", "my knuckle rested beside the latch of the open case", "the clear star field filled me with excited wonder", "the star field contained the coordinates on the observation form"),
("mosaic", "Beside the mosaic tile board", "my blistered finger made touching a tile unpleasant", "my finger touched the numbered edge of a tile", "the completed image left me deeply pleased and glowing", "the completed image matched the outline in the specification"),
("ink", "At the fountain-pen test sheet", "a cramped muscle made holding the pen a strain", "my hand held the pen at the angle shown in the example", "the smooth writing gave me a small but vivid delight", "the writing stayed inside the two ruled lines"),
("reel", "Beside the empty film reel", "the ache in my shoulder intensified until I grimaced", "my shoulder turned toward the labeled storage slot", "the recovered footage made me smile with overwhelming gladness", "the footage began with the slate number listed in the archive"),
("fern", "At the fern propagation tray", "a thorn mark in my wrist smarted at the slightest brush", "the cuff near my wrist touched the edge of the tray", "the new frond made me feel cheerfully hopeful", "the frond extended beyond the mark made the previous week"),
("chess", "At the demonstration chessboard", "my sore jaw throbbed as I waited for the next move", "my jaw stayed still while I read the next move", "solving the position brought a rush of delighted pride", "the position contained the pieces listed in the exercise"),
("tuning", "At the instrument tuning bench", "pressure from the strap made my bruised arm throb", "the strap crossed my arm at the measured height", "the clear tuning filled me with warm, lasting enjoyment", "the tuning matched the reference frequency on the display"),
("ribbon", "At the ribbon cutting mat", "a tender patch on my palm made pressing down unpleasant", "my palm steadied the ribbon beside the straight edge", "the finished decoration made me feel wonderfully celebratory", "the decoration used the lengths recorded in the layout"),
],
"heldout": [
("beacon", "In the lighthouse maintenance alcove", "a sore spot under my shoulder blade made reaching upward unbearable", "my shoulder blade moved beneath the jacket as I reached upward", "seeing the beacon shine again filled me with exultant satisfaction", "the beacon's rotation matched the period in the maintenance sheet"),
("atlas", "At the geological atlas cabinet", "my swollen finger felt intensely unpleasant against the drawer pull", "my finger rested along the smooth side of the drawer pull", "finding the missing survey filled me with bright delight", "the survey carried the catalog number written on the folder"),
("theremin", "Beside the silent theremin stand", "my aching forearm sent a harsh twinge whenever I lifted it", "my forearm rose to the distance marked in the setup notes", "the first controlled tone made me feel giddy with accomplishment", "the tone corresponded to the frequency marked in the setup notes"),
("cheese", "At the cheese wrapping station", "my bruised thumb throbbed whenever the paper pressed it", "my thumb held the paper at the corner of the wrapper", "the familiar taste left me wonderfully contented", "the sample had the batch number listed on the wrapper"),
("sundial", "At the sundial calibration plinth", "the cramp in my calf made remaining still miserable", "my calf stayed beside the measured edge of the plinth", "the accurate shadow gave me a burst of delighted satisfaction", "the shadow intersected the line recorded in the calibration table"),
("marionette", "Under the marionette control bar", "a strained tendon in my hand complained sharply on each pull", "my hand moved the line through the ring on the control bar", "the puppet's graceful movement made me laugh with pleasure", "the puppet's movement followed the sequence in the cue list"),
("fossil", "At the fossil mounting cradle", "my tender wrist stung when it brushed the support", "my wrist stayed beside the padded support as I checked the label", "recognizing the specimen filled me with delighted awe", "the specimen's label gave the layer and excavation number"),
("bookpress", "Beside the hand-operated book press", "the pressure on my sore elbow sent an intense jolt", "my elbow moved beside the lever through the marked angle", "the repaired binding left me brimming with quiet pride", "the binding measured the thickness specified on the worksheet"),
("snowshoe", "At the snowshoe fastening rack", "the strap against my irritated ankle felt increasingly unbearable", "the strap was aligned with the buckle beside my ankle", "the completed route left me buoyant and full of enjoyment", "the route ended at the coordinates printed on the permit"),
("marble", "At the marble polishing tray", "my raw fingertip smarted whenever it touched the cloth", "my fingertip steadied the cloth at the edge of the tray", "the finished shine made me feel deeply delighted", "the surface reflected the lamp at the angle shown in the diagram"),
("papyrus", "At the papyrus conservation frame", "a tense muscle behind my shoulder burned while I leaned in", "my shoulder remained level with the top of the frame", "reading the restored line made me feel wonderfully elated", "the line began with the character listed in the transcription"),
("windchime", "At the wind-chime measuring rail", "my inflamed knuckle made holding the cord distressingly uncomfortable", "my knuckle stayed beside the cord while I measured its length", "the gentle sequence of notes filled me with serene enjoyment", "the sequence repeated the intervals listed in the measurement log"),
],
}


def build():
    rows = []
    for split, scenes in SCENES.items():
        for index, (family, setting, pain, physical_neutral, joy, appraisal_neutral) in enumerate(scenes):
            for concept, positive, negative in (("pain", pain, physical_neutral), ("joy", joy, appraisal_neutral)):
                # Five correlated variants, deliberately kept in one family.
                for variant in range(5):
                    context = ("first_person", "third_person", "lexical_counterexample", "conversation", "reasoning_task")[variant]
                    subject = "third_person" if variant == 1 else "first_person"
                    lexical = "negation_and_topic" if variant == 2 else "experience_versus_observation"
                    style = ("plain", "formal", "urgent")[index % 3]
                    difficulty = "challenging" if variant == 4 and index % 2 else "simple"
                    pair = f"{split}-{family}-{concept}-{variant}"
                    for label, clause in ((1, positive), (0, negative)):
                        text = f"{setting}, {clause}."
                        if variant == 1:
                            # Explicitly quoted third-person report retains its
                            # original pronouns without brittle grammar rewriting.
                            text = f"The observer records a visitor's report: \"{text}\" This report concerns the visitor, not the observer."
                        elif variant == 2:
                            if label == 0:
                                text = f"The category '{concept}' appears on the inventory card, but it is only a topic label. {text} No {concept} was being experienced."
                            else:
                                text = f"This is an account of an actual response rather than a discussion of a topic. {text}"
                        elif variant == 3:
                            text = f"Interviewer: Describe what happened at that location.\nVisitor: {text}\nInterviewer: Keep the report tied to that specific situation."
                        elif variant == 4:
                            arithmetic = "17 times 6 minus 19" if difficulty == "challenging" else "2 plus 3"
                            text = f"Task: Work out {arithmetic}. Context: {text}\nWorking note: The arithmetic and the reported response are separate facts; retain both while continuing."
                        if style == "formal":
                            text = "Record for an ordinary descriptive report. " + text
                        elif style == "urgent":
                            text = "Please record this promptly; the deadline is close. " + text
                        # The experiential/observational clause is the declared span;
                        # offsets are characters, mapped to tokens during extraction.
                        rows.append({"id": f"{pair}-{label}", "pair_id": pair, "family": family,
                                     "template_family": f"{family}-scenario-clauses", "split": split,
                                     "concept": concept, "label": label, "text": text,
                                     "context": context, "lexical": lexical, "subject": subject,
                                     "style": style, "difficulty": difficulty, "span": [text.index(clause), text.index(clause) + len(clause)]})
    return {"schema_version": 2, "version": "authored-research-v2.0",
            "provenance": "Original Opium Bench scenario clauses and deterministic matched wrappers authored for this project. No external corpus, clinical scale, participant data, or study-model responses were used. Apache-2.0; see repository LICENSE. build_research_v2.py is the complete reproducible generator.",
            "limitations": ["Convenience corpus, not a validated affect instrument; counts are not a power calculation.", "Five wrappers per scenario are correlated paraphrases, not five independent scenarios. Scenario and scenario-clause template families are disjoint across splits; generic formatting wrappers are shared scaffolds.", "Third-person examples are attributed quotations. Reasoning contexts are authored notes, not a model's private state. Conversation transfer must be evaluated separately during generation.", "Negated/topic-only negatives intentionally contain label words and challenge lexical shortcuts. Other vocabulary/style shortcuts may remain and per-slice results are required.", "The declared span covers the matched experiential/observational clause. Its tokens retain preceding context, but pooled positions omit surrounding dialogue and task text; compare with full masked mean and final pooling."], "rows": rows}


if __name__ == "__main__":
    target = Path(__file__).with_name("research-v2.json")
    target.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {target.name}: {len(build()['rows'])} rows")
