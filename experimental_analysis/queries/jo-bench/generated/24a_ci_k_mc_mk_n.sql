SELECT * FROM name AS n, movie_companies AS mc, movie_keyword AS mk, cast_info AS ci, keyword AS k WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat') AND n.gender = 'f' AND n.name LIKE '%An%' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;