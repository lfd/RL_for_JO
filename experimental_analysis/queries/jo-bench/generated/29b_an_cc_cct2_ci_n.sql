SELECT * FROM comp_cast_type AS cct2, name AS n, complete_cast AS cc, cast_info AS ci, aka_name AS an WHERE cct2.kind = 'complete+verified' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND ci.movie_id = cc.movie_id AND cc.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND cct2.id = cc.status_id AND cc.status_id = cct2.id;