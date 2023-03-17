SELECT * FROM name AS n, char_name AS chn, aka_name AS an, movie_companies AS mc, cast_info AS ci WHERE chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND n.gender = 'f' AND n.name LIKE '%An%' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND n.id = an.person_id AND an.person_id = n.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id;