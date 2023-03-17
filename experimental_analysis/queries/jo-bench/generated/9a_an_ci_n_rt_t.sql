SELECT * FROM name AS n, aka_name AS an, cast_info AS ci, role_type AS rt, title AS t WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%Ang%' AND rt.role = 'actress' AND t.production_year BETWEEN 2005 AND 2015 AND ci.movie_id = t.id AND t.id = ci.movie_id AND ci.role_id = rt.id AND rt.id = ci.role_id AND n.id = ci.person_id AND ci.person_id = n.id AND an.person_id = n.id AND n.id = an.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;