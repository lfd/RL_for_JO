SELECT * FROM title AS t, char_name AS chn, cast_info AS ci WHERE chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2005 AND t.id = ci.movie_id AND ci.movie_id = t.id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;