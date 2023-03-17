SELECT * FROM cast_info AS ci, aka_name AS an1, name AS n1, movie_companies AS mc WHERE ci.note = '(voice: English version)' AND mc.note LIKE '%(Japan)%' AND mc.note NOT LIKE '%(USA)%' AND n1.name LIKE '%Yo%' AND n1.name NOT LIKE '%Yu%' AND an1.person_id = n1.id AND n1.id = an1.person_id AND n1.id = ci.person_id AND ci.person_id = n1.id AND an1.person_id = ci.person_id AND ci.person_id = an1.person_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id;