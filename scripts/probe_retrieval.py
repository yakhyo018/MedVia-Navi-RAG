"""Print the top hits for a few questions — a quick look before the real evaluation."""

import sys

from common import script_settings

from navi.container import build_container

QUESTIONS = sys.argv[1:] or [
    'How much does Laser Toning cost at Seoul Glow Dermatology?',
    'Сколько стоит отбеливание зубов в Itaewon Compass Dental?',
    'Bronni qancha vaqt oldin bekor qilsa bo‘ladi?',
    'Is an interpreter free?',
    'Do you offer LASIK eye surgery in Busan?',
    'What is the weather in Seoul tomorrow?',
]

c = build_container(script_settings(min_score=0.0))
for q in QUESTIONS:
    print(f'\n{q}')
    for h in c.rag.retrieve(q, 3):
        print(f'  {h.score:.3f}  {h.heading[:70]}')
c.store.close()
