SELECT * FROM name AS n, cast_info AS ci, title AS t, char_name AS chn WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND t.production_year > 2010 AND t.title LIKE 'Kung Fu Panda%' AND t.id = ci.movie_id AND ci.movie_id = t.id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;