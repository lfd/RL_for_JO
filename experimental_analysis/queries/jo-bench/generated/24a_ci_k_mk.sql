SELECT * FROM keyword AS k, cast_info AS ci, movie_keyword AS mk WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat') AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND k.id = mk.keyword_id AND mk.keyword_id = k.id;