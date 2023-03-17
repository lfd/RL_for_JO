SELECT * FROM person_info AS pi, cast_info AS ci, title AS t, aka_name AS an WHERE an.name LIKE '%a%' AND pi.note = 'Volker Boehm' AND t.production_year BETWEEN 1980 AND 1984 AND t.id = ci.movie_id AND ci.movie_id = t.id AND pi.person_id = an.person_id AND an.person_id = pi.person_id AND pi.person_id = ci.person_id AND ci.person_id = pi.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;