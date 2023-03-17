SELECT * FROM cast_info AS ci, char_name AS chn, movie_companies AS mc, name AS n, title AS t, aka_name AS an WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND ci.movie_id = t.id AND t.id = ci.movie_id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND an.person_id = n.id AND n.id = an.person_id AND an.person_id = ci.person_id AND ci.person_id = an.person_id;