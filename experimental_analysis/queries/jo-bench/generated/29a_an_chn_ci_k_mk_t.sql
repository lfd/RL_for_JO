SELECT * FROM keyword AS k, movie_keyword AS mk, title AS t, cast_info AS ci, aka_name AS an, char_name AS chn WHERE chn.name = 'Queen' AND ci.note IN ('(voice)', '(voice) (uncredited)', '(voice: English version)') AND k.keyword = 'computer-animation' AND t.title = 'Shrek 2' AND t.production_year BETWEEN 2000 AND 2010 AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mk.movie_id AND mk.movie_id = t.id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND ci.person_id = an.person_id AND an.person_id = ci.person_id AND chn.id = ci.person_role_id AND ci.person_role_id = chn.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;